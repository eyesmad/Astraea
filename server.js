// Import the express module
const express = require('express');

// Initialize the Express application instance
const app = express();
const PORT = 3000;

// Define a basic route handler for HTTP GET requests
app.get('/', (req, res) => {
  res.send('Hello World!');
});

// Start the server and listen on port 3000
app.listen(PORT, () => {
  console.log(`Server is running at http://localhost:${PORT}`);
});