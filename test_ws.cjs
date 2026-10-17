const WebSocket = require('ws');
const ws = new WebSocket('ws://localhost:3002');
ws.on('open', () => {
    console.log("Connected to Python directly");
    ws.send(JSON.stringify({type: 'config', sampleRate: 32000}));
});
ws.on('message', (d) => console.log("Recv:", d.toString()));
ws.on('close', () => console.log("Closed"));
ws.on('error', console.error);
