const WebSocket = require('ws');
const ws = new WebSocket('ws://localhost:3002');
ws.on('open', () => {
    console.log("Connected");
    setInterval(() => {
        const buf = Buffer.alloc(1288);
        buf.writeDoubleLE(Date.now(), 0);
        ws.send(buf);
    }, 20); // 50 times a second
    
    setInterval(() => {
        ws.send(JSON.stringify({type: 'face', t: Date.now(), p: new Array(100).fill(0)}));
    }, 33); // 30 times a second
});
ws.on('message', (d) => console.log("Recv:", d.toString()));
ws.on('close', () => console.log("Closed"));
ws.on('error', console.error);
