# Redis Setup

## Starting Redis

In WSL terminal:

```bash
redis-server
```

This starts Redis in foreground mode on `localhost:6379`.

## Stopping Redis

Press `Ctrl+C` in the terminal running `redis-server`.

## Background Mode (Optional)

To run Redis as a daemon:

```bash
redis-server --daemonize yes
```

To stop:

```bash
redis-cli shutdown
```

## Verify Connection

```bash
redis-cli ping
# Should return: PONG
```
