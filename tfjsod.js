
module.exports = function (RED) {
    function TensorFlowObjDet(n) {
        var fs = require('fs');
        var path = require('path');
        var jpeg = require('jpeg-js');
        var express = require("express");
        var pureimage = require("pureimage");
        var compression = require("compression");
        var tf = require('@tensorflow/tfjs-node');
        var cocoSsd = require('@tensorflow-models/coco-ssd');

        /* suggestion from https://github.com/tensorflow/tfjs/issues/2029 */
        const nodeFetch = require('node-fetch'); // <<--- ADD
        global.fetch = nodeFetch; // <<--- ADD
        /* ************************************************************** */

        RED.nodes.createNode(this, n);
        this.model = n.model;
        this.modelUrlType = n.modelUrlType;
        this.scoreThreshold = n.scoreThreshold;
        this.maxDetections = n.maxDetections;
        this.passthru = n.passthru || "false";
        this.modelUrl = n.modelUrl || undefined; // "http://localhost:1880/coco/model.json"
        this.lineColour = n.lineColour || "magenta";
        var node = this;
        var model;

        RED.httpNode.use(compression());
        RED.httpNode.use('/models', express.static(__dirname + '/models'));

        function getDuration() {
            // Calculate the execution time (in milliseconds)
            var stopTime = process.hrtime(node.startTime);
            var duration = (stopTime[0] * 1000000000 + stopTime[1]) / 1000000;

            // Start a new counter
            node.startTime = process.hrtime();

            return Math.round(duration);
        }

        async function loadFont() {
            node.fnt = pureimage.registerFont(path.join(__dirname,'./SourceSansPro-Regular.ttf'),'Source Sans Pro');
            node.fnt.load();
        }
        loadFont();

        async function loadModel() {
            node.status({fill:'yellow', shape:'ring', text:'Loading model...'});
            // TODO add the labels file (https://raw.githubusercontent.com/google-coral/test_data/master/coco_labels.txt) to Dave's repository instead of the labels array below??
            //labels = fs.readFileSync('./model/labels.txt', 'utf8').split('\n');
            node.labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'n/a', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'n/a', 'backpack', 'umbrella', 'n/a', 'n/a', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'n/a', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'n/a', 'dining table', 'n/a', 'n/a', 'toilet', 'n/a', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'n/a', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'];

            // Initialize the model
            switch(node.model) {
            case "coco_ssd_mobilenet_v2":
                if (node.modelUrlType === "local") {
                    node.modelUrl = "http://localhost:"+RED.settings.uiPort+RED.settings.httpNodeRoot+"models/coco-ssd/model.json";
                }
                model = await cocoSsd.load({modelUrl: node.modelUrl});
                break;

            case "coco_ssd_mobilenet_v2_coral":
                // This is not a great fix for thing not working on Mac - but at least keeps the other options working
                var tflite = require('tfjs-tflite-node');
                var {CoralDelegate} = require('coral-tflite-delegate');
                if (node.modelUrlType === "local") {
                    // Coral edge TPU models available at https://coral.ai/models/
                    //node.modelUrl = "https://raw.githubusercontent.com/google-coral/test_data/master/tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite";
                    node.modelUrl = "http://localhost:"+RED.settings.uiPort+RED.settings.httpNodeRoot+"models/coco-coral/tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite"
                }
                // Use a Coral delegate, which makes use of the EdgeTPU Runtime Library (libedgetpu)..
                // When no delegate is specified, the model would be processed by the CPU.
                model = await tflite.loadTFLiteModel(node.modelUrl, {delegates: [new CoralDelegate()]});
                break;

            case "yolo_ssd_v5":
                if (node.modelUrlType === "local") {
                    node.modelUrl = "http://localhost:"+RED.settings.uiPort+RED.settings.httpNodeRoot+"models/yolo5n/model.json";
                }
                else if (node.modelUrl === undefined) { node.modelUrl = DEFAULT_MODEL_LOCATION; }
                console.log("Yolo URL",node.modelUrl)
                model = await tf.loadGraphModel(node.modelUrl);
                console.log("LOADED Yolo");
                break;
            }

            node.ready = true;
            node.status({fill:'green', shape:'dot', text:'Model ready'});
        }

        loadModel();

        async function getImage(msg) {
            fetch(msg.payload)
                .then(r => r.buffer())
                .then(buf => msg.payload = buf)
                .then(function() {handleMsg(msg) });
        }

        async function handleMsg(msg) {
            if (node.passthru === "true") { msg.image = msg.payload; }

            node.startTime = process.hrtime();
            msg.executionTimes = {};
            msg.executionTimes.decoding = getDuration();
            msg.maxDetections = msg.maxDetections || node.maxDetections || 40;

            // Decode the image and convert it to a tensor (in this case it will become 3D tensors based on the encoded bytes).
            // The tfjs image decoding supports BMP, GIF, JPEG and PNG.
            try {
                var imageTensor = tf.node.decodeImage(msg.payload);
            }
            catch(e) {
                node.error("Payload does not seem to be a valid image buffer.",msg)
                return;
            }

            var modelWidth, modelHeight, input;

            var makeBoxes = function(boxes, scores, classes, count) {
                var imageHeight = imageTensor.shape[0];
                var imageWidth = imageTensor.shape[1];
                for (var i = 0; i < count; i++) {
                    // var score = scores[i];
                    // if(score >= node.scoreThreshold) { // TODO reuse the code from Dave
                    const clazz = node.labels[classes[i]];
                    // only report those that we have in the labels array.
                    if (clazz) {
                        var x1 = parseInt(Math.max(0, boxes[i*4+0] * imageWidth));
                        var y1 = parseInt(Math.max(0, boxes[i*4+1] * imageHeight));
                        var x2 = parseInt(Math.min(imageWidth, boxes[i*4+2] * imageWidth));
                        var y2 = parseInt(Math.min(imageHeight, boxes[i*4+3] * imageHeight));

                        msg.payload.push({
                            class: clazz,
                            bbox: [x1, y1, Math.abs(x2 - x1), Math.abs(y2 - y1)], // Denormalized bbox with format [x_min, y_min, width, height]
                            score: scores[i] * 100
                        })
                    }
                }
            }

            switch(node.model) {
            case "coco_ssd_mobilenet_v2":
                // The tfjs models work out-of-the-box since all Tensor handling is done inside the tfjs models.
                // This in contradiction to the TfLite model below, in which case this node needs to do all input/output processing on its own...
                // The resulting bounding boxes will have format [x_min, y_min, width, height].
                msg.payload = await model.detect(imageTensor, msg.maxDetections);
                break;

            case "coco_ssd_mobilenet_v2_coral":
                var resizedImageWidth = 300;
                var resizedImageHeight = 300;

                // This model has been trained on images of size 300x300, which means our input images also need to have that same size.
                // Otherwise the prediction will throw an exception: Input tensor shape mismatch: expect '1,300,300,3', got '1,480,640,3'.
                // We will use the 'neirest neighbour' algorithm, because that is the fastest one (which still delivers enough accuracy).
                var resizedImageTensor = tf.image.resizeNearestNeighbor(imageTensor, [resizedImageHeight, resizedImageWidth]);

                // Transform the 3D image into a 4D tensor that the TfLite model requires.
                // Otherwise the prediction will throw following exception: Input tensor shape mismatch: expect '1,300,300,3', got '300,300,3'.
                resizedImageTensor = tf.expandDims(resizedImageTensor, 0);

                // It is NOT required to normalize the input images (i.e. from values in the range [0, 255] to [0,1]).
                // Because this model (which is a TfLite model converted for Coral TPU's) requires uint8 values, in contradiction to its original models.
                // The model throws an exception if we normalize the input images: "Data type mismatch: input tensor expects 'uint8', got 'float32'"

                // Run the object detection on the image.
                var prediction = model.predict(resizedImageTensor);

                // Parse the output from the object detection.  Each model will produce different output.
                // For example here https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2 you can see which date each tensor in output result contains.
                var bboxes = prediction["StatefulPartitionedCall:3;StatefulPartitionedCall:2;StatefulPartitionedCall:1;StatefulPartitionedCall:0"].arraySync()[0];
                var classes = prediction["StatefulPartitionedCall:3;StatefulPartitionedCall:2;StatefulPartitionedCall:1;StatefulPartitionedCall:01"].arraySync()[0];
                var scores = prediction["StatefulPartitionedCall:3;StatefulPartitionedCall:2;StatefulPartitionedCall:1;StatefulPartitionedCall:02"].arraySync()[0];
                var objectCount = prediction["StatefulPartitionedCall:3;StatefulPartitionedCall:2;StatefulPartitionedCall:1;StatefulPartitionedCall:03"].arraySync()[0];

                // TODO check whether it is usefull to call tf.image.nonMaxSuppressionAsync() ???
                // See here how this algorithm works: https://www.analyticsvidhya.com/blog/2020/08/selecting-the-right-bounding-box-using-non-max-suppression-with-implementation/

                // Get the original image dimensions from the unresized 3D tensor
                var imageHeight = imageTensor.shape[0];
                var imageWidth = imageTensor.shape[1];

                msg.payload = [];

                for(var i = 0; i < objectCount; i++) {
                    var score = scores[i];
                    if(score >= node.scoreThreshold) { // TODO reuse the code from Dave
                        var clazz = classes[i];

                        // The normalized bbox has the format [ymin, xmin, ymax, xmax]
                        var normalizedBbox = bboxes[i];

                        // Denormalization can be executed by multiplying with the image width and height.
                        // The advantage of normalized bounding boxes, is that it fits both the resized and original images (which is send in the output msg).
                        // The prediction can give coordinates outside of the image dimensions, so force them to be withing the image (via min and max).
                        var x1 = parseInt(Math.max(0, normalizedBbox[1] * imageWidth));
                        var y1 = parseInt(Math.max(0, normalizedBbox[0] * imageHeight));
                        var x2 = parseInt(Math.min(imageWidth, normalizedBbox[3] * imageWidth));
                        var y2 = parseInt(Math.min(imageHeight, normalizedBbox[2] * imageHeight));

                        msg.payload.push({
                            class: node.labels[clazz],
                            bbox: [x1, y1, Math.abs(x2 - x1), Math.abs(y2 - y1)], // Denormalized bbox with format [x_min, y_min, width, height]
                            score: scores[i] * 100
                        })
                    }
                }
                break;

            case "yolo_ssd_v5":
                [modelWidth, modelHeight] = model.inputs[0].shape.slice(1, 3);
                input = tf.tidy(() => {
                    return tf.image
                        .resizeBilinear(imageTensor, [modelWidth, modelHeight])
                        .div(255.0)
                        .expandDims(0);
                });

                msg.payload = [];
                await model.executeAsync(input).then((res) => {
                    const [boxes, scores, classes, rc] = res.slice(0, 4);
                    const boxes_data = boxes.dataSync();
                    const scores_data = scores.dataSync();
                    const classes_data = classes.dataSync();
                    const rc_data = rc.dataSync();
                    // console.log("BD",boxes_data)
                    // console.log("SD",scores_data)
                    // console.log("CD",classes_data)
                    // console.log("RC",rc_data[0])
                    // res.forEach(t => t.print());
                    makeBoxes(boxes_data, scores_data, classes_data, rc_data[0]);
                });
                console.log("PAY",msg.payload)
                break;
            }

            msg.executionTimes.detection = getDuration();

            msg.shape = imageTensor.shape;
            msg.classes = {};
            msg.scoreThreshold = msg.scoreThreshold || node.scoreThreshold || 0.5;

            // TODO add filtering based on minimum and maximum bbox size.
            for (var i=0; i<msg.payload.length; i++) {
                if (msg.payload[i].score < msg.scoreThreshold) {
                    msg.payload.splice(i,1);
                    i = i - 1;
                }
                if (msg.payload[i].hasOwnProperty("class")) {
                    msg.classes[msg.payload[i].class] = (msg.classes[msg.payload[i].class] || 0 ) + 1;
                }
            }

            tf.dispose(imageTensor);

            // Draw bounding boxes on the image
            if (node.passthru === "bbox") {
                var jimg;

                if (node.passthru === "bbox") { jimg = jpeg.decode(msg.image); }
                var pimg = pureimage.make(jimg.width,jimg.height);
                var ctx = pimg.getContext('2d');
                var scale = parseInt((jimg.width + jimg.height) / 500 + 0.5);
                ctx.bitmap.data = jimg.data;
                for (var k=0; k<msg.payload.length; k++) {
                    ctx.fillStyle = node.lineColour;
                    ctx.strokeStyle = node.lineColour;
                    ctx.font = scale*8+"pt 'Source Sans Pro'";
                    ctx.fillText(msg.payload[k].class, msg.payload[k].bbox[0] + 4, msg.payload[k].bbox[1] - 4)
                    ctx.lineWidth = scale;
                    ctx.lineJoin = 'bevel';
                    ctx.rect(msg.payload[k].bbox[0], msg.payload[k].bbox[1], msg.payload[k].bbox[2], msg.payload[k].bbox[3]);
                    ctx.stroke();
                }

                msg.executionTimes.drawing = getDuration();
                msg.image = jpeg.encode(pimg,70).data;
                msg.executionTimes.encoding = getDuration();
            }

            node.send(msg);
        }

        node.on('input', function (msg) {
            try {
                if (node.ready) {
                    msg.image = msg.payload;
                    if (typeof msg.payload === "string") {
                        if (msg.payload.startsWith("http")) {
                            getImage(msg);
                            return;
                        }
                        else if (msg.payload.startsWith("data:image/jpeg")) {
                            msg.payload = Buffer.from(msg.payload.split(";base64,")[1], 'base64');
                        }
                        else { msg.payload = fs.readFileSync(msg.payload); }
                    }
                    // Allow repalcing the label dynamically
                    // Also if any are missing from the array then don't report/draw boxes later
                    if (Array.isArray(msg.labels)) { node.labels = msg.labels; }
                    if (typeof msg.labels === "string") { node.labels = msg.labels.split(','); }
                    if (msg.payload) {
                        handleMsg(msg);
                    }
                }
            } catch (error) {
                node.error(error, msg);
            }
        });

        node.on("close", function () {
            node.status({});
            node.ready = false;
            model.dispose();
            model = null;
            node.fnt = null;
        });
    }
    RED.nodes.registerType("tensorflowObjDet", TensorFlowObjDet);
};
