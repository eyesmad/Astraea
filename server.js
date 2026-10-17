// Import the express module
import express from 'express';
import { WebSocketServer } from 'ws';

// Initialize the Express application instance
const app = express();
const PORT = 3000;
const WSS_PORT = 3001;

// WEBSOCKET ============================================================================
const wss = new WebSocketServer({ port: WSS_PORT });

console.log(`WebSocket server is running on ws://localhost:${WSS_PORT}`);

wss.on('connection', (socket) => {
  console.log(`New client connected. Total clients: ${wss.clients.size}`);

  // Listen for messages from this specific client
  socket.on('message', (data, isBinary) => {
    console.log(`Received message: ${data}`);

    // Broadcast the message to all other connected clients
    wss.clients.forEach((client) => {
      if (client !== socket && client.readyState === WebSocket.OPEN) {
        client.send(data, { binary: isBinary });
      }
    });
  });

  // Handle client disconnection
  socket.on('close', () => {
    console.log(`Client disconnected. Total clients: ${wss.clients.size}`);
  });

  // Handle connection errors
  socket.on('error', (error) => {
    console.error(`Socket error: ${error.message}`);
  });
})

// STANDARD HTTP ROUTES ===================================================================

// Define a basic route handler for HTTP GET requests
app.get('/', (req, res) => {
  res.send('Hello World!');
});

// Start the server and listen on port 3000
app.listen(PORT, '0.0.0.0', () => {
  console.log(`Server is running at http://localhost:${PORT}`);
});