
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
        var tflite = require('tfjs-tflite-node');
        var {CoralDelegate} = require('coral-tflite-delegate');

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
        RED.httpNode.use('/coco', express.static(__dirname + '/models/coco-ssd'));

        async function loadFont() {
            node.fnt = pureimage.registerFont(path.join(__dirname,'./SourceSansPro-Regular.ttf'),'Source Sans Pro');
            node.fnt.load();
        }
        loadFont();
        
        async function loadModel() {
            node.status({fill:'yellow', shape:'ring', text:'Loading model...'});
        
            // Initialize the model
            switch(node.model) {
                case "coco_ssd_mobilenet_v2":
                    if (node.modelUrlType === "local") {
                        node.modelUrl = "http://localhost:"+RED.settings.uiPort+RED.settings.httpNodeRoot+"coco/model.json";
                    }
                    model = await cocoSsd.load({modelUrl: node.modelUrl});
                    break;
                case "coco_ssd_mobilenet_v2_coral":
                    if (node.modelUrlType === "local") {
                        // Coral edge TPU models available at https://coral.ai/models/
                        // TODO add the file to Dave's repository
                        node.modelUrl = "https://raw.githubusercontent.com/google-coral/test_data/master/tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite";
                    }

                    // Use a Coral delegate, which makes use of the EdgeTPU Runtime Library (libedgetpu)..  
                    // When no delegate is specified, the model would be processed by the CPU.
                    const options = {delegates: [new CoralDelegate()]};
                    model = await tflite.loadTFLiteModel(node.modelUrl, options);
                    // TODO add the labels file (https://raw.githubusercontent.com/google-coral/test_data/master/coco_labels.txt) to Dave's repository instead of the labels array below?? 
                    //labels = fs.readFileSync('./model/labels.txt', 'utf8').split('\n');
                    node.labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'n/a', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'n/a', 'backpack', 'umbrella', 'n/a', 'n/a', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'n/a', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'n/a', 'dining table', 'n/a', 'n/a', 'toilet', 'n/a', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'n/a', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'];
                    break;
                case "yolo_ssd_v5":
                
                    break;
            }
            
            node.ready = true;
            node.status({fill:'green', shape:'dot', text:'Model ready'});
        }
        
        loadModel();

        async function getImage(m) {
            fetch(m.payload)
                .then(r => r.buffer())
                .then(buf => m.payload = buf)
                .then(function() {handleMsg(m) });
        }

        async function handleMsg(m) {
            if (node.passthru === "true") { m.image = m.payload; }
            
            // Decode the image and convert it to a tensor (in this case it will become 3D tensors based on the encoded bytes). 
            // The tfjs image decoding supports BMP, GIF, JPEG and PNG.
            var imageTensor = tf.node.decodeImage(m.payload);

            m.maxDetections = m.maxDetections || node.maxDetections || 40;
            
            switch(node.model) {
                case "coco_ssd_mobilenet_v2":
                    // The tfjs models work out-of-the-box since all Tensor handling is done inside the tfjs models.
                    // This in contradiction to the TfLite model below, in which case this node needs to do all input/output processing on its own...
                    // The resulting bounding boxes will have format [x_min, y_min, width, height].
                    m.payload = await model.detect(imageTensor, m.maxDetections);
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
                    
                    m.payload = [];

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

                            m.payload.push({
                                class: node.labels[clazz],
                                bbox: [x1, y1, Math.abs(x2 - x1), Math.abs(y2 - y1)], // Denormalized bbox with format [x_min, y_min, width, height]
                                score: scores[i] * 100
                            })
                        }
                    }
                    break;
                case "yolo_ssd_v5":
                
                    break;
            }

            m.shape = imageTensor.shape;
            m.classes = {};
            m.scoreThreshold = m.scoreThreshold || node.scoreThreshold || 0.5;

            // TODO add filtering based on minimum and maximum bbox size.
            for (var i=0; i<m.payload.length; i++) {
                if (m.payload[i].score < m.scoreThreshold) {
                    m.payload.splice(i,1);
                    i = i - 1;
                }
                if (m.payload[i].hasOwnProperty("class")) {
                    m.classes[m.payload[i].class] = (m.classes[m.payload[i].class] || 0 ) + 1;
                }
            }

            tf.dispose(imageTensor);

            // Draw bounding boxes on the image
            if (node.passthru === "bbox") {
                var jimg;

                if (node.passthru === "bbox") { jimg = jpeg.decode(m.image); }
                var pimg = pureimage.make(jimg.width,jimg.height);
                var ctx = pimg.getContext('2d');
                var scale = parseInt((jimg.width + jimg.height) / 500 + 0.5);
                ctx.bitmap.data = jimg.data;
                for (var k=0; k<m.payload.length; k++) {
                    ctx.fillStyle = node.lineColour;
                    ctx.strokeStyle = node.lineColour;
                    ctx.font = scale*8+"pt 'Source Sans Pro'";
                    ctx.fillText(m.payload[k].class, m.payload[k].bbox[0] + 4, m.payload[k].bbox[1] - 4)
                    ctx.lineWidth = scale;
                    ctx.lineJoin = 'bevel';
                    ctx.rect(m.payload[k].bbox[0], m.payload[k].bbox[1], m.payload[k].bbox[2], m.payload[k].bbox[3]);
                    ctx.stroke();
                }
                m.image = jpeg.encode(pimg,70).data;
            }

            node.send(m);
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
                    
                    handleMsg(msg);
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
