# Fragaria Frontend

This is the frontend interface for Fragaria, an advanced Chain of Thought (CoT) Reasoning API with Reinforcement Learning (RL).

## Overview

The frontend provides a user-friendly web interface to interact with the Fragaria API. It allows users to:

- Submit problems for analysis using natural language
- View the Chain of Thought reasoning process
- See the final results in a clean, readable format

## Getting Started

### Prerequisites

- Node.js (v14 or higher)
- npm (comes with Node.js)

### Installation

1. Install the dependencies:
   ```bash
   npm install
   ```

2. Build the frontend:
   ```bash
   npm run build
   ```

### Development

To run the development server with hot reloading:
```bash
npm run dev
```

Navigate to [localhost:8080](http://localhost:8080) to view the application.

### Production

To create an optimized production build:
```bash
npm run build
```

To serve the production build:
```bash
npm run start
```

## Integration with Fragaria API

The frontend is designed to work seamlessly with the Fragaria backend API. When you run the Fragaria server, it will automatically serve the frontend at the root path.

To start the Fragaria server with the frontend:
```bash
fragaria-server
```

Or directly with Python:
```bash
python -m fragaria.main
```

The API will be available at `http://localhost:8000` by default, with the frontend served at the root path.

## Architecture

The frontend is built with Svelte and communicates with the Fragaria API through HTTP requests. It features:

- A clean, responsive design
- Real-time display of the reasoning process
- Error handling and user feedback
- Mobile-friendly interface

## Customization

You can customize the frontend by modifying the Svelte components in the `src` directory. The main entry point is `src/main.js`, and the main component is `src/App.svelte`.

## Deployment

To deploy the frontend, you need to:

1. Build the frontend: `npm run build`
2. Start the Fragaria server: `fragaria-server`

The Fragaria server will automatically serve the built frontend files.

## Contributing

Contributions to improve the frontend are welcome. Please follow the main Fragaria contributing guidelines.

## License

This frontend is part of the Fragaria project and is licensed under the MIT License. See the main project LICENSE file for details.