package com.example.users;

import org.springframework.web.bind.annotation.*;

/**
 * REST endpoints for user management.
 */
@RestController
@RequestMapping("/api/users")
public class UserController {

    private final UserService service;

    public UserController(UserService service) {
        this.service = service;
    }

    @GetMapping("/{id}")
    public User get(@PathVariable Long id) {
        return service.find(id);
    }

    @PostMapping
    public User create(@RequestBody UserPayload payload) {
        return service.create(payload);
    }
}
