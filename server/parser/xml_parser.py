from __future__ import annotations

import xml.etree.ElementTree as ET

from server.parser.base import CodeSymbol


def _strip_ns(tag: str) -> str:
    """Remove XML namespace: {http://...}localname → localname"""
    return tag.split("}")[-1] if "}" in tag else tag


def _find_line(lines: list[str], *candidates: str) -> int:
    for candidate in candidates:
        for i, line in enumerate(lines, 1):
            if candidate in line:
                return i
    return 1


def _child_text(elem: ET.Element, local_tag: str) -> str:
    child = next((c for c in elem if _strip_ns(c.tag) == local_tag), None)
    return (child.text or "").strip() if child is not None else ""


def _find_children(elem: ET.Element, local_tag: str) -> list[ET.Element]:
    return [c for c in elem if _strip_ns(c.tag) == local_tag]


def _make_doc(filename: str, text: str, file_path: str, total_lines: int) -> CodeSymbol:
    return CodeSymbol(
        name=filename,
        symbol_type="document",
        language="xml",
        source=text,
        file_path=file_path,
        start_line=1,
        end_line=total_lines,
        signature=filename,
    )


def _parse_pom(
    root: ET.Element,
    text: str,
    lines: list[str],
    file_path: str,
    filename: str,
    total_lines: int,
) -> list[CodeSymbol]:
    symbols: list[CodeSymbol] = []

    group_id = _child_text(root, "groupId")
    artifact_id = _child_text(root, "artifactId")
    version = _child_text(root, "version")
    proj_name = f"{group_id}:{artifact_id}" if group_id and artifact_id else filename
    sig = f"{proj_name}:{version}" if version else proj_name

    symbols.append(CodeSymbol(
        name=proj_name,
        symbol_type="project",
        language="xml",
        source=text,
        file_path=file_path,
        start_line=1,
        end_line=total_lines,
        signature=sig,
        extras={"groupId": group_id, "artifactId": artifact_id, "version": version},
    ))

    # Collect dependency containers: <dependencies> and <dependencyManagement><dependencies>
    dep_containers: list[tuple[ET.Element, bool]] = []  # (elem, is_managed)
    for child in root:
        tag = _strip_ns(child.tag)
        if tag == "dependencies":
            dep_containers.append((child, False))
        elif tag == "dependencyManagement":
            for gc in child:
                if _strip_ns(gc.tag) == "dependencies":
                    dep_containers.append((gc, True))

    for container, managed in dep_containers:
        for dep in _find_children(container, "dependency"):
            g = _child_text(dep, "groupId")
            a = _child_text(dep, "artifactId")
            v = _child_text(dep, "version")
            scope = _child_text(dep, "scope")
            dep_name = f"{g}:{a}" if g and a else a or g or "dependency"
            dep_sig = dep_name
            if v:
                dep_sig += f":{v}"
            if scope:
                dep_sig += f" [{scope}]"
            line = _find_line(lines, f">{a}<", f">{g}<")
            symbols.append(CodeSymbol(
                name=dep_name,
                symbol_type="dependency",
                language="xml",
                source=ET.tostring(dep, encoding="unicode"),
                file_path=file_path,
                start_line=line,
                end_line=line,
                signature=dep_sig,
                extras={"groupId": g, "artifactId": a, "version": v, "scope": scope, "managed": managed},
            ))

    # Plugins
    plugins_elem: ET.Element | None = None
    for child in root:
        if _strip_ns(child.tag) == "build":
            for gc in child:
                if _strip_ns(gc.tag) == "plugins":
                    plugins_elem = gc
                    break
                if _strip_ns(gc.tag) == "pluginManagement":
                    for ggc in gc:
                        if _strip_ns(ggc.tag) == "plugins":
                            plugins_elem = ggc
                            break

    if plugins_elem is not None:
        for plugin in _find_children(plugins_elem, "plugin"):
            g = _child_text(plugin, "groupId")
            a = _child_text(plugin, "artifactId")
            v = _child_text(plugin, "version")
            plugin_name = f"{g}:{a}" if g and a else a or g or "plugin"
            plugin_sig = f"{plugin_name}:{v}" if v else plugin_name
            line = _find_line(lines, f">{a}<", f">{g}<")
            symbols.append(CodeSymbol(
                name=plugin_name,
                symbol_type="plugin",
                language="xml",
                source=ET.tostring(plugin, encoding="unicode"),
                file_path=file_path,
                start_line=line,
                end_line=line,
                signature=plugin_sig,
                extras={"groupId": g, "artifactId": a, "version": v},
            ))

    return symbols


def _parse_spring_beans(
    root: ET.Element,
    text: str,
    lines: list[str],
    file_path: str,
    filename: str,
    total_lines: int,
) -> list[CodeSymbol]:
    symbols: list[CodeSymbol] = []

    for bean in _find_children(root, "bean"):
        attrs = {_strip_ns(k): v for k, v in bean.attrib.items()}
        bean_id = attrs.get("id", "")
        bean_class = attrs.get("class", "")
        short_class = bean_class.rsplit(".", 1)[-1] if bean_class else ""
        bean_name = bean_id or short_class or "bean"
        sig = f'<bean id="{bean_id}" class="{bean_class}">' if bean_id else f'<bean class="{bean_class}">'
        search = f'id="{bean_id}"' if bean_id else f'class="{bean_class}"'
        line = _find_line(lines, search)
        symbols.append(CodeSymbol(
            name=bean_name,
            symbol_type="bean",
            language="xml",
            source=ET.tostring(bean, encoding="unicode"),
            file_path=file_path,
            start_line=line,
            end_line=line,
            signature=sig,
            extras=attrs,
        ))

    return symbols


def _parse_generic(
    root: ET.Element,
    text: str,
    lines: list[str],
    file_path: str,
    filename: str,
    total_lines: int,
) -> list[CodeSymbol]:
    symbols: list[CodeSymbol] = []
    children = list(root)
    include_all = len(children) <= 20

    for child in children:
        tag = _strip_ns(child.tag)
        attrs = {_strip_ns(k): v for k, v in child.attrib.items()}
        elem_id = attrs.get("id") or attrs.get("name")
        if not elem_id and not include_all:
            continue
        elem_name = elem_id or f"<{tag}>"
        attr_preview = " ".join(f'{k}="{v}"' for k, v in list(attrs.items())[:3])
        sig = f"<{tag}" + (f" {attr_preview}" if attr_preview else "") + ">"
        search_terms: list[str] = []
        if "id" in attrs:
            search_terms.append(f'id="{attrs["id"]}"')
        elif "name" in attrs:
            search_terms.append(f'name="{attrs["name"]}"')
        search_terms.append(f"<{tag}")
        line = _find_line(lines, *search_terms)
        symbols.append(CodeSymbol(
            name=elem_name,
            symbol_type="element",
            language="xml",
            source=ET.tostring(child, encoding="unicode"),
            file_path=file_path,
            start_line=line,
            end_line=line,
            signature=sig,
            extras={"tag": tag, **attrs},
        ))

    return symbols


class XmlParser:
    def supported_extensions(self) -> list[str]:
        return [".xml"]

    def language(self) -> str:
        return "xml"

    def parse_file(self, source: bytes, file_path: str) -> list[CodeSymbol]:
        text = source.decode("utf-8", errors="replace")
        lines = text.splitlines()
        filename = file_path.rsplit("/", 1)[-1]
        total_lines = len(lines) or 1

        try:
            root = ET.fromstring(source)
        except ET.ParseError:
            return [_make_doc(filename, text, file_path, total_lines)]

        root_tag = _strip_ns(root.tag)

        if filename == "pom.xml" or root_tag == "project":
            symbols = _parse_pom(root, text, lines, file_path, filename, total_lines)
        elif root_tag == "beans":
            symbols = _parse_spring_beans(root, text, lines, file_path, filename, total_lines)
        else:
            symbols = _parse_generic(root, text, lines, file_path, filename, total_lines)

        return symbols or [_make_doc(filename, text, file_path, total_lines)]
