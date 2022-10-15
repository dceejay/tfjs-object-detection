
module.exports = function (RED) {
    function TensorFlowObjDet(n) {
        var fs = require('fs');
        var path = require('path');
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
        this.scoreThreshold = n.scoreThreshold;
        this.maxDetections = n.maxDetections;
        this.passthru = n.passthru || "false";
        this.outputFormat = n.outputFormat;
        this.lineColor = n.lineColor || "magenta";
        this.colors = [];
        this.tensorflowConfig = n.tensorflowConfig;
        this.tensorflowConfigNode = null;
        this.model = null;

        var node = this;

        if (node.tensorflowConfig) {
            // Retrieve the config node, where the model is configured
            node.tensorflowConfigNode = RED.nodes.getNode(node.tensorflowConfig);
        }

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
            var modelUrl;
            
            node.status({fill:'yellow', shape:'ring', text:'Loading labels...'});
 
            try {
                // Load the labels, to identify the type of selected object (e.g. person, car, ...)
                switch(node.tensorflowConfigNode.labelsType) {
                case "path":
                    var labelsPath = node.tensorflowConfigNode.labels;

                    if(!path.isAbsolute(labelsPath)) {
                        labelsPath= path.resolve(__dirname, labelsPath);
                    }

                    var labelsBuffer = fs.readFileSync(labelsPath);
                    node.labels = labelsBuffer.toString().split(/\r?\n/);
                    break;
                case "url":
                    var response = await nodeFetch(node.tensorflowConfigNode.labels);
                    if(!response.ok) {
                        throw response.statusText;
                    }

                    var body = response.getBody();
                    node.labels = body.toString().split(/\r?\n/);
                    break;
                case "json":
                    node.labels = JSON.parse(node.tensorflowConfigNode.labels);
                    break;
                }
                
                if(!Array.isArray(node.labels)) {
                    throw "The content is not an array";
                }
            }
            catch(err) {
                node.status({fill:'red', shape:'ring', text:'Failed to load labels'});
                node.error("Can't load labels: " + err);
                node.labels = null;
                return;
            }

            node.status({fill:'yellow', shape:'ring', text:'Loading colors...'});

            try {
                // Load the colors, one for every type of selected object (e.g. person=red, car=green, ...)
                switch(node.tensorflowConfigNode.colorsType) {
                case "path":
                    var colorsPath = node.tensorflowConfigNode.colors;

                    if(!path.isAbsolute(colorsPath)) {
                        labelsPath= path.resolve(__dirname, colorsPath);
                    }

                    var colorsBuffer = fs.readFileSync(colorsPath);
                    node.colors = colorsBuffer.toString().split(/\r?\n/);
                    break;
                case "url":
                    var response = await nodedeFetch(node.tensorflowConfigNode.colors);
                    if(!response.ok) {
                        throw response.statusText;
                    }

                    var body = response.getBody();
                    node.labels = body.toString().split(/\r?\n/);
                    break;
                case "json":
                    node.colors = JSON.parse(node.tensorflowConfigNode.colors);
                    break;
                }
                
                // The colors are optional, but when they are specified they should be correct
                if(node.colors) {
                    if(!Array.isArray(node.colors)) {
                        throw "The content is not an array";
                    }
                    
                    if(node.colors.length > 0 && node.colors.length != node.labels.length) {
                        throw "The number of colors (" + node.colors.length + " does not match the number of labels (" + node.labels.length + ")";
                    }
                }
            }
            catch(err) {
                node.status({fill:'red', shape:'ring', text:'Failed to load colors'});
                node.error("Can't load colors: " + err);
                node.colors = null;
                // Non blocking...
                //return;
            }

            node.status({fill:'yellow', shape:'ring', text:'Loading model...'});

            switch(node.tensorflowConfigNode.modelType) {
            case "path":
                // TODO this currently only works for RELATIVE paths e.g. "models/coco-ssd/model.json"
                modelUrl = "http://localhost:" + RED.settings.uiPort + RED.settings.httpNodeRoot + node.tensorflowConfigNode.model;
                break;
            case "url":
                modelUrl = node.tensorflowConfigNode.model;
                break;
            }

            console.log("Loading the " + node.tensorflowConfigNode.runtime + " model from " + node.tensorflowConfigNode.model);

            // Initialize the model
            switch(node.tensorflowConfigNode.runtime) {
            case "tfjs":
                cocoSsd.load({modelUrl: modelUrl}).then(model => {
                    node.model = model;
                    node.ready = true;
                    node.status({fill:'green', shape:'dot', text:'Tfjs model ready'});
                }).catch(err => {
                    node.status({fill:'red', shape:'ring', text:'Failed to load model'});
                    node.error("Can't load tfjs model: ", err);
                });

                break;

            case "tflite":
                // This is not a great fix for thing not working on Mac - but at least keeps the other options working
                try {
                    var tflite = require('tfjs-tflite-node');
                    var {CoralDelegate} = require('coral-tflite-delegate');
                }
                catch(e) {
                    node.status({fill:'red', shape:'ring', text:'Failed to load TFLite library'});
                    node.error("Failed to load TFLite library",e);
                    return;
                }

                //TODO: check whether the TPU runtime has been installed

                // Use a Coral delegate, which makes use of the EdgeTPU Runtime Library (libedgetpu)..
                // When no delegate is specified, the model would be processed by the CPU.
                tflite.loadTFLiteModel(modelUrl, {delegates: [new CoralDelegate()]}).then(model => {
                    node.model = model;
                    node.ready = true;
                    node.status({fill:'green', shape:'dot', text:'TFLite model ready'});
                }).catch(err => {
                    node.status({fill:'red', shape:'ring', text:'Failed to load model'});
                    node.error("Can't load TFLite model: ", err);
                });
                
                break;

            case "graph":
                tf.loadGraphModel(modelUrl).then(model => {
                    node.model = model;
                    node.ready = true;
                    node.status({fill:'green', shape:'dot', text:'Graph model ready'});
                }).catch(err => {
                    node.status({fill:'red', shape:'ring', text:'Failed to load model'});
                    node.error("Can't load graph model: ", err);
                });

                break;
            }
        }

        if (node.tensorflowConfigNode) {
            loadModel();
        }

        async function getImage(msg) {
            fetch(msg.payload)
                .then(r => r.buffer())
                .then(buf => msg.payload = buf)
                .then(function() {handleMsg(msg) });
        }

        function getOutputProperty(output, name, property, propertyType) {
            var propertyTensor;

            // The 'property' can be a property name or a property index
            switch(propertyType) {
                case "index":
                    var outputValues = Object.values(output);
                    
                    if(property >= outputValues.length) {
                        throw "Cannot get " + name + " from the detection result, because the index " + property + " is out of range";
                    }
                    
                    propertyTensor = outputValues[property]; // N-th value
                    break;
                case "name":
                    if(!output.hasOwnProperty(property)) {
                        throw "Cannot get " + name + " from the detection result, because the property " + property + " does not exist";
                    }
                
                    propertyTensor = output[property];
                    break;
            }

            return propertyTensor.arraySync()[0];
        }

        async function handleMsg(msg) {
            var resizedImageTensor;

            if (node.passthru === "true") { msg.image = msg.payload; }

            node.startTime = process.hrtime();
            msg.executionTimes = {};
            msg.maxDetections = msg.maxDetections || node.maxDetections || 40;

            // ---------------------------------------------------------------------------------------------------------
            // 1. Decode the input image to raw pixels (and store it inside a tensor)
            // ---------------------------------------------------------------------------------------------------------

            // Decode the image and convert it to a tensor (in this case it will become 3D tensors based on the encoded bytes).
            // The tfjs image decoding supports BMP, GIF, JPEG and PNG.
            try {
                var imageTensor = tf.node.decodeImage(msg.payload);
            }
            catch(er) {
                node.error("Error while decoding input image: " + err);
                return;
            }

            msg.executionTimes.decoding = getDuration();

            // ---------------------------------------------------------------------------------------------------------
            // 2. Prepare the input image
            // ---------------------------------------------------------------------------------------------------------

            var modelWidth, modelHeight;

            // Get the image dimensions expected by the model, which corresponds to the dimensions of the images that have been used to train the model.
            // These dimensions are stored in the model shape tensor, which contains the following information: [batchsize, width, height, colorCount]
            if(node.tensorflowConfigNode.runtime === "tfjs") {
                // The tfjs model contains of a sub-model instance
                modelWidth = node.model.model.inputs[0].shape[1];
                modelHeight = node.model.model.inputs[0].shape[2];
            }
            else {
                modelWidth = node.model.inputs[0].shape[1];
                modelHeight = node.model.inputs[0].shape[2];
            }
            
            // Get the dimensions of the input image
            var imageHeight = imageTensor.shape[0];
            var imageWidth = imageTensor.shape[1];
            
            // Preprocessing is NOT required for runtime type 'tfjs', because the tfjs model has the image preprocessing code inside the tfjs model
            if(node.tensorflowConfigNode.runtime === "tfjs") {
                resizedImageTensor = imageTensor;
            }
            else {
                // Resize the input image if it's dimensions don't match with the model requirement
                if(imageHeight != modelHeight || imageWidth != modelWidth) {
                    // Otherwise the prediction will throw an exception.  For example: Input tensor shape mismatch: expect '1,300,300,3', got '1,480,640,3'.
                    switch(node.tensorflowConfigNode.resizeAlgorithm) {
                    case "bilinear":
                        // The 'bilinear' algorithm offers smooth transitions.
                        resizedImageTensor = tf.image.resizeBilinear(imageTensor, [modelWidth, modelHeight]);
                        break;
                    case "neirest":
                        // The 'neirest neighbour' algorithm is the fastest one (which still delivers enough accuracy).
                        resizedImageTensor = tf.image.resizeNearestNeighbor(imageTensor, [modelHeight, modelWidth]);
                        break;
                    }
                }
                
                if(node.model.inputs[0].dtype === 'float32' && resizedImageTensor.dtype === 'int32') {
                    // When the model requires dtype int32, all tensor elements contain RGB colors in the range from 0 to 255.
                    // When the model requires dtype int32, all tensor elements need to be normalized (to the range from 0 to 1).
                    // To accomplish that, divide all tensor elements by 255
                    resizedImageTensor = resizedImageTensor.div(255.0);
                }
 
                msg.executionTimes.resizing = getDuration();

                // Transform the 3D image into a 4D tensor that the TfLite/Graph model requires.
                // Otherwise the prediction will throw an exception.  For example: Input tensor shape mismatch: expect '1,300,300,3', got '300,300,3'.
                resizedImageTensor = tf.expandDims(resizedImageTensor, 0);
            }

            // ---------------------------------------------------------------------------------------------------------
            // 3. Execute object detection on the image
            // ---------------------------------------------------------------------------------------------------------
            
            var detectionResult;

            switch(node.tensorflowConfigNode.runtime) {
            case "tfjs":
                // The tfjs models are pretty easy to use, since all Tensor handling is done inside the tfjs models.
                // Which means the input is an image, and the output is bboxes/scores/...
                // This in contradiction to the other models (TfLite, Graph, ...), where all input/output processing needs to be done by this node.
                // TODO don't use await to avoid uncaught exceptions
                detectionResult = await node.model.detect(imageTensor, msg.maxDetections);
                break;

            case "tflite":
                // It is NOT required to normalize the input images (i.e. from values in the range [0, 255] to [0,1]).
                // Because this model (which is a TfLite model converted for Coral TPU's) requires uint8 values, in contradiction to its original models.
                // The model throws an exception if we normalize the input images: "Data type mismatch: input tensor expects 'uint8', got 'float32'"

                detectionResult = node.model.predict(resizedImageTensor);
                break;

            case "graph":
                // TODO can predict also be used here instead of executeAsync?
                await node.model.executeAsync(resizedImageTensor).then(result => {
                    detectionResult = result
                }).catch(err => {
                    throw err;
                });
                break;
            }

            msg.executionTimes.detection = getDuration();

            // ---------------------------------------------------------------------------------------------------------
            // 4. Parse the object detection output (tensors)
            // ---------------------------------------------------------------------------------------------------------

            msg.shape = imageTensor.shape;
            msg.classes = {};
            msg.scoreThreshold = msg.scoreThreshold || node.scoreThreshold || 0.5;
            
            // Postprocessing is NOT required for runtime type 'tfjs', because the tfjs model has the postprocessing code inside the tfjs model
            if(node.tensorflowConfigNode.runtime === "tfjs") {
                msg.payload = detectionResult;
            }
            else {
                msg.payload = [];

                // Get the required information from the specified output tensors
                var bboxes      = getOutputProperty(detectionResult, "bboxes"      , node.tensorflowConfigNode.bboxProperty , node.tensorflowConfigNode.bboxPropertyType);
                var classes     = getOutputProperty(detectionResult, "classes"     , node.tensorflowConfigNode.classProperty, node.tensorflowConfigNode.classPropertyType);
                var scores      = getOutputProperty(detectionResult, "scores"      , node.tensorflowConfigNode.scoreProperty, node.tensorflowConfigNode.scorePropertyType);
                var objectCount = getOutputProperty(detectionResult, "object count", node.tensorflowConfigNode.countProperty, node.tensorflowConfigNode.countPropertyType);

                for(var i = 0; i < objectCount; i++) {
                    var score = scores[i];

                    if(score >= node.scoreThreshold) { // TODO reuse the code from Dave
                        var classIndex = classes[i];

                        // The normalized bbox has the format [ymin, xmin, ymax, xmax]
                        var normalizedBbox = bboxes[i];
                        
                        var x1, y1, x2, y2;
                        
                        // Determine the bbox upper-left and lower-right corner points, based on the specified model output bbox format
                        switch(node.tensorflowConfigNode.bboxFormat) {
                        case "[x1,y1,x2,y2]":
                            x1 = normalizedBbox[0];
                            y1 = normalizedBbox[1];
                            x2 = normalizedBbox[2];
                            y2 = normalizedBbox[3];
                            break;
                        case "[y1,x1,y2,x2]":
                            x1 = normalizedBbox[1];
                            y1 = normalizedBbox[0];
                            x2 = normalizedBbox[3];
                            y2 = normalizedBbox[2];
                            break;
                        case "[x1,y1,w,h]":
                            x1 = normalizedBbox[0];
                            y1 = normalizedBbox[1];
                            x2 = normalizedBbox[2] - normalizedBbox[0];
                            y2 = normalizedBbox[3] - normalizedBbox[1];
                            break;
                        case "[y1,x1,h,w]":
                            x1 = normalizedBbox[1];
                            y1 = normalizedBbox[0];
                            x2 = normalizedBbox[3] - normalizedBbox[1];
                            y2 = normalizedBbox[2] - normalizedBbox[0];
                            break;
                        }

                        // Denormalization can be executed by multiplying with the image width and height.
                        // The advantage of normalized bounding boxes, is that it fits both the resized and original images (which is send in the output msg).
                        x1 *= imageWidth;
                        x2 *= imageWidth;
                        y1 *= imageHeight;
                        y2 *= imageHeight;
                        
                        // The detection result can contain coordinates outside of the image dimensions, so force them to be withing the image (via min and max).
                        var x1 = Math.max(0, x1);
                        var y1 = Math.max(0, y1);
                        var x2 = Math.min(imageWidth, x2);
                        var y2 = Math.min(imageHeight, y2);
                        
                        // Calculate the bbox dimensions
                        var width = Math.abs(x2 - x1);
                        var height = Math.abs(y2 - y1);
                        
                        // For all runtime types we will send the bbox in the format [x_min, y_min, width, height]
                        var denormalizedBbox = [x1, y1, width, height];

                        msg.payload.push({
                            classIndex: classIndex,
                            class: node.labels[classIndex],
                            bbox: denormalizedBbox,
                            score: scores[i]
                        })
                    }
                }
            }

            // TODO add filtering based on minimum and maximum bbox size.

            // Sort the array so we get highest scores, to make sure the highest scores are preserved in case the array length is trimmed afterward
            msg.payload.sort((a, b) => (a.score < b.score) ? 1 : -1)
            
            for (var j=0; j < Math.min(msg.payload.length, msg.maxDetections); j++) {
                // TODO use a reduce function for this
                if (msg.payload[j].score < msg.scoreThreshold) {
                    msg.payload.splice(j,1);
                    j = j - 1;
                }
                
                // TODO in a separate loop??
                if (msg.payload[j] && msg.payload[j].hasOwnProperty("class")) {
                    msg.classes[msg.payload[j].class] = (msg.classes[msg.payload[j].class] || 0 ) + 1;
                }
            }

            // ---------------------------------------------------------------------------------------------------------
            // 5. Draw the bounding boxes on top of the raw image (if required)
            // ---------------------------------------------------------------------------------------------------------

            if (node.passthru === "bbox") {
                var rgbaTensor;

                if (imageTensor.shape[2] === 3) {
                    // When the image tensor contains RGB (i.e. 3 channels), a fourth alpha channel needs to be added.
                    // PureImage requires RGBA data to display inside it's canvas, so no RGB data is allowed.
                    // This is similar to the html canvas (see https://developer.mozilla.org/en-US/docs/Web/API/ImageData/data).
                    // So add an alpha channel to the image if it is missing.  This will be the case for decoded jpeg's, since jpeg has no transparency support.
                    // We have asked the Tensorflow team to add alpha channel support for jpeg (see https://github.com/tensorflow/tfjs/issues/6911).
                    var alphaChannelTensor = tf.fill([imageHeight, imageWidth, 1], 255, 'int32');
                    rgbaTensor = tf.concat([imageTensor, alphaChannelTensor], 2);
                }
                else {
                    // The image tensor contains RGBA (i.e. 4 channels), when it's shape is [height, width, 4]).
                    rgbaTensor = imageTensor;
                }

                // Get the (decoded) input image data from the image tensor
                var rawImage = rgbaTensor.dataSync(); 

                msg.executionTimes.rgb_to_rgba = getDuration();

                // Create a new empty PureImage 2D canvas
                var pimg = pureimage.make(imageWidth, imageHeight);
                var ctx = pimg.getContext('2d');
                var scale = parseInt((imageWidth + imageHeight) / 500 + 0.5);

                // Store the raw image inside the PureImage 2D canvas
                ctx.bitmap.data = rawImage;

                // Draw all the bounding boxes in the PureImage 2D canvas
                msg.payload.forEach(function (detectedObject, index) {
                    var bboxColor;

                    if(node.colors && Array.isArray(node.colors) && node.colors.length > 0) {
                        // Lookup the index of the class to cross ref into color table
                        bboxColor = node.colors[detectedObject.classIndex];
                    }
                    else {
                        // Apply the specified default color
                        bboxColor = node.lineColor;
                    }

                    ctx.fillStyle = bboxColor;
                    ctx.strokeStyle = bboxColor;
                    ctx.font = scale*8+"pt 'Source Sans Pro'";
                    ctx.fillText(detectedObject.class, detectedObject.bbox[0] + 4, detectedObject.bbox[1] - 4)
                    ctx.lineWidth = scale;
                    ctx.lineJoin = 'bevel';
                    ctx.rect(detectedObject.bbox[0], detectedObject.bbox[1], detectedObject.bbox[2], detectedObject.bbox[3]);
                    ctx.stroke();
                });

                msg.executionTimes.drawing = getDuration();
                
                // Convert the annotated image (from the PureImage canvas) to an rgba tensor
                var annotatedImageTensor = tf.tensor3d(pimg.data, [imageHeight, imageWidth, 4], 'int32');
                
                // TODO the output image format (jpeg, raw, ...) should be adjustable in the config screen
                
                // The encodeJpeg does not support an alpha channel, so convert the annotated image tensor from RGBA to RGB
                var channels = tf.split(annotatedImageTensor, 4, 2);            
                var rgbTensor = tf.concat([channels[0], channels[1], channels[2]], 2);
                
                // Override the original image tensor by the one containing the bounding box drawings
                tf.dispose(imageTensor);
                imageTensor = rgbTensor;

                msg.executionTimes.rgba_to_rgb = getDuration();
            }
                
            // ---------------------------------------------------------------------------------------------------------
            // 6. Encode the raw (annotated) image to JPEG
            // ---------------------------------------------------------------------------------------------------------
            
            if(node.outputFormat == "jpg") {
                // Encode the RGB tensor to a JPEG image
                var jpeg = await tf.node.encodeJpeg(imageTensor);
                
                // Convert the jpeg image from Uint8Array to a buffer
                msg.image.data = Buffer.from(jpeg);
                msg.image.type = "jpg";
            }
            else {
                var imageArray.data = imageTensor.dataSync();

                // Convert the raw image from Uint32Array to a buffer
                msg.image = Buffer.from(imageArray);
                msg.image.type = "raw";
            }
            
            msg.image.width = imageWidth;
            msg.image.height = imageHeight

            msg.executionTimes.encoding = getDuration();

            // Cleanup all tensors, to avoid memory leakage
            tf.dispose([annotatedImageTensor, alphaChannelTensor, rgbaTensor]);

            tf.dispose(imageTensor);

            // Calculate the total execution time
            msg.executionTimes.total = 0;
            Object.values(msg.executionTimes).forEach(function (executionTime, index) {
                msg.executionTimes.total += executionTime;
            });

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
                    // Allow replacing the labels array dynamically - either as an array or a csv string
                    // Also if any are missing from the array then don't report/draw boxes later
                    if (Array.isArray(msg.labels)) { node.labels = msg.labels; }
                    if (typeof msg.labels === "string") { node.labels = msg.labels.split(','); }
                    if (Array.isArray(msg.colors)) { node.colors = msg.colors; }
                    if (typeof msg.colors === "string") { node.colors = msg.colors.split(','); }
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
            node.model.dispose();
            node.model = null;
            node.fnt = null;
        });
    }
    RED.nodes.registerType("tensorflowObjDet", TensorFlowObjDet);
};