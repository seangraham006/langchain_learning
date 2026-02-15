# TODO

- Review whether `ChronicleAgent` should be implemented as a singleton.
- Review making a Redis handler context manager that auto-closes the connection (e.g., `with redis_handler:`).
- Review functionality for dynamic agent spawning — allow agents (e.g., spawner agent) to create and add new agent processes to the townhall at runtime (e.g., create a Farmer agent). Consider human-in-the-loop approval for spawning decisions.
