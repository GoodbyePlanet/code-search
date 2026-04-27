// Package routing wires up the HTTP handlers.
package routing

import "net/http"

// Router dispatches incoming requests.
type Router struct {
	mux *http.ServeMux
}

// Handler is implemented by every request handler.
type Handler interface {
	Serve(w http.ResponseWriter, r *http.Request)
}

// NewRouter constructs a Router with default routes.
func NewRouter() *Router {
	return &Router{mux: http.NewServeMux()}
}

// Handle registers a handler for the given pattern.
func (r *Router) Handle(pattern string, h Handler) {
	r.mux.Handle(pattern, http.HandlerFunc(h.Serve))
}
