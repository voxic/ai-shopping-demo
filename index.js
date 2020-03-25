/*
 * Project: ai-shopping-demo
 * File: index.js
 * Author: Emil Nilsson
 * Contact: emil.nilsson@nutanix.com
 */

let express = require('express');
let app = express();
const server = require('http').Server(app);
const NATS = require('nats') // Require the NATS module
const io = require('socket.io')(server);
const isBase64 = require('is-base64');


// Create a connection to the NATS demo server
console.log("Trying to connect to NATS at " + process.env.NATS_PORT_4222_TCP);
const nc = NATS.connect(process.env.NATS_PORT_4222_TCP); 


app.set('view engine', 'ejs');

//set upp public directory to serve static files
app.use(express.static('public'));


//When we get a socket.io connection
io.on('connection', function (socket) {
    // Subscribe to the NATS topic
    console.log("Subbed to: " + process.env.NATS_ENDPOINT);
    nc.subscribe(process.env.NATS_ENDPOINT, (msg) => {
        try {
            var json_payload = JSON.parse(String(msg).substring(String(msg).indexOf("{"))) //Extract the JSON part of the NATS message

            //Extract the image part of the JSON message
            if("frame" in json_payload){
                if(json_payload.frame.endsWith('\n')){
                    json_payload.frame = json_payload.frame.replace(/\n/gm, '')
                }
                console.log(json_payload);

                //Check to see that is valid base64 encoding before sending to front-end
                if(isBase64(json_payload.frame)){
                    socket.emit('frame', json_payload.frame);
                    socket.emit('data', JSON.stringify(json_payload.json_payload));
                }
                else {
                    console.log("Not base64")
                }
            }else {
                socket.emit('data', JSON.stringify(json_payload.json_payload));
            }

        }
        catch(err){
            console.log("Bad JSON " + err)
        }
    })
});


app.get('/', (req, res) => {
  res.render('index');
});

server.listen(4000);
console.log("Service running on port 4000");