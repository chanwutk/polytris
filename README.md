# Polyis: Efficient Object Tracking through Relevant Pixel Compression

## Development Setup

To set up this repository for development, follow these steps.

1.  **Configure `docker-compose.yml`:**

    Edit `docker-compose.yml` to avoid conflicts with other users.

    -   **Change the container name:** Modify `container_name` from `polyis` to a unique name, such as `polyis-<your_username>`.
    -   **Change data and cache directories:** Update the volume paths for your data and cache directories. Use a path you have access to, like `/data/<your_username>`.

    ```yaml
    # docker-compose.yml
    services:
      polyis:
        build: .
        container_name: polyis-<your_username>
        ...
        volumes:
          - .:/polyis
          # Change to your local path
          - /data/<your_username>/data/polyis-data:/polyis-data
          - /data/<your_username>/data/polyis-cache:/polyis-cache
          ...
    ```

2.  **Update the `dock` script:**

    If you changed the `container_name` in `docker-compose.yml`, you must also update the `dock` script to match it.

    ```bash
    #! /bin/bash
    # dock

    docker exec -it polyis-<your_username> bash
    ```

3.  **Start the Docker Development Environment:**

    Run the following command to build and start the container in detached mode:

    ```bash
    docker compose up --detach --build
    ```

4.  **Access the Development Container:**

    You can access the running container in two ways:

    -   **VSCode/Cursor:** Use the "Remote - SSH" and "Remote - Containers" extensions to attach your editor to the running container.
    -   **Terminal:** Run the `./dock` script to access the shell inside the container.
