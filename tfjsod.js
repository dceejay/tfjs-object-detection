
module.exports = function (RED) {
    function TensorFlowObjDet(config) {
        var fs = require('fs');
        var path = require('path');
        var express = require("express");
        var pureimage = require("pureimage");
        var compression = require("compression");
        var tf = require('@tensorflow/tfjs-node');
        var cocoSsd = require('@tensorflow-models/coco-ssd');
        var process = require('process');

        /* suggestion from https://github.com/tensorflow/tfjs/issues/2029 */
        const nodeFetch = require('node-fetch'); // <<--- ADD
        global.fetch = nodeFetch; // <<--- ADD
        /* ************************************************************** */

        RED.nodes.createNode(this, config);
        this.minimumScore = parseFloat(config.minimumScore || 0);
        this.classFilter = config.classFilter || "[]";
        this.classFilterType = config.classFilterType;
        this.minimumHeight = parseInt(config.minimumHeight || 0);
        this.maximumHeight = parseInt(config.maximumHeight || Number.MAX_SAFE_INTEGER);
        this.minimumWidth = parseInt(config.minimumWidth || 0);
        this.maximumWidth = parseInt(config.maximumWidth || Number.MAX_SAFE_INTEGER);
        this.minimumAspectRatio = parseFloat(config.minimumAspectRatio || 0);
        this.maximumAspectRatio = parseFloat(config.maximumAspectRatio || Number.MAX_SAFE_INTEGER);
        this.maximumDetections = parseInt(config.maximumDetections || Number.MAX_SAFE_INTEGER);
        this.inputFormat = config.inputFormat;
        this.passthru = config.passthru || "none";
        this.annotationExpression = config.annotationExpression;
        this.annotationBackground = config.annotationBackground;
        this.bboxPadding = parseInt(config.bboxPadding || 0);
        this.outputFormat = config.outputFormat;
        this.lineColor = config.lineColor || "magenta";
        this.colors = [];
        this.tensorflowConfig = config.tensorflowConfig;
        this.tensorflowConfigNode = null;
        this.model = null;
        this.busy = false;
        this.skippedMsgCount = 0;

        var node = this;
        
        try {
            // The config contains the array as a string (e.g. '["person", "car"]')
            node.classFilter = JSON.parse(node.classFilter);
        }
        catch(err) {
            node.warn("The 'class' filter property should contain an array of strings, e.g. [\"person\", \"car\"]");
        }

        if (node.tensorflowConfig) {
            // Retrieve the config node, where the model is configured
            node.tensorflowConfigNode = RED.nodes.getNode(node.tensorflowConfig);
        }

        RED.httpNode.use(compression());
        RED.httpNode.use('/models', express.static(__dirname + '/models'));
        
        // TensorflowJS chooses the best backend for processing automatically
        console.log("Tfjs backend: " + tf.getBackend());

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
        
        function modelLoaded(type, model) {
            node.model = model;
            node.status({fill: 'green', shape: 'dot', text: type + ' model ready'});
      
            // Determine the model input metadata, i.e. information about which kind of input it expects
            if (node.tensorflowConfigNode.runtime === "tfjs") {
                // The tfjs model contains of a sub-model instance
                node.modelInputMetadata = node.model.model.inputs[0];
            }
            else {
                node.modelInputMetadata = node.model.inputs[0];
            }

            // The 'idle' time starts from the moment the model has been loaded
            node.startTime = process.hrtime();
        }

        async function loadModel() {
            var modelUrl;

            node.status({fill:'yellow', shape:'ring', text:'Loading labels...'});

            try {
                // Load the labels, to identify the type of selected object (e.g. person, car, ...)
                switch(node.tensorflowConfigNode.labelsType) {
                    case "path":
                        var labelsPath = node.tensorflowConfigNode.labels;

                        if (!path.isAbsolute(labelsPath)) {
                            labelsPath= path.resolve(__dirname, labelsPath);
                        }

                        var labelsBuffer = fs.readFileSync(labelsPath);
                        node.labels = labelsBuffer.toString().split(/\r?\n/);
                        break;
                    case "url":
                        var response = await nodeFetch(node.tensorflowConfigNode.labels);
                        if (!response.ok) {
                            throw response.statusText;
                        }

                        const data = await response.text();
                        node.labels = data.split(/\r?\n/);
                        break;
                    case "json":
                        node.labels = JSON.parse(node.tensorflowConfigNode.labels);
                        break;
                }

                if (!Array.isArray(node.labels)) {
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

                        if (!path.isAbsolute(colorsPath)) {
                            labelsPath= path.resolve(__dirname, colorsPath);
                        }

                        var colorsBuffer = fs.readFileSync(colorsPath);
                        node.colors = colorsBuffer.toString().split(/\r?\n/);
                        break;
                    case "url":
                        var response = await nodeFetch(node.tensorflowConfigNode.colors);
                        if (!response.ok) {
                            throw response.statusText;
                        }

                        const data = await response.text();
                        node.labels = data.split(/\r?\n/);
                        break;
                    case "json":
                        node.colors = JSON.parse(node.tensorflowConfigNode.colors);
                        break;
                }

                // The colors are optional, but when they are specified they should be correct
                if (node.colors) {
                    if (!Array.isArray(node.colors)) {
                        throw "The content is not an array";
                    }

                    if (node.colors.length > 0 && node.colors.length != node.labels.length) {
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
                        modelLoaded("Tfjs", model);
                    }).catch(err => {
                        node.status({fill:'red', shape:'ring', text:'Failed to load model'});
                        node.error("Can't load tfjs model: " + err);
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
                        modelLoaded("TFLite", model);
                    }).catch(err => {
                        node.status({fill:'red', shape:'ring', text:'Failed to load model'});
                        node.error("Can't load TFLite model: " + err);
                    });

                    break;

                case "graph":
                    // console.log("URL:",modelUrl)
                    tf.loadGraphModel(modelUrl).then(model => {
                        modelLoaded("Graph", model);
                    }).catch(err => {
                        node.status({fill:'red', shape:'ring', text:'Failed to load model'});
                        node.error("Can't load graph model: " + err);
                    });

                break;
            }
        }

        if (node.tensorflowConfigNode) {
            loadModel();
        }

        function getOutputProperty(output, name, property, propertyType) {
            var propertyTensor;

            // The 'property' can be a property name or a property index
            switch(propertyType) {
                case "index":
                    var outputValues = Object.values(output);

                    if (property >= outputValues.length) {
                        throw "Cannot get " + name + " from the detection result, because the index " + property + " is out of range";
                    }

                    propertyTensor = outputValues[property]; // N-th value
                    break;
                case "name":
                    if (!output.hasOwnProperty(property)) {
                        throw "Cannot get " + name + " from the detection result, because the property " + property + " does not exist";
                    }

                    propertyTensor = output[property];
                    break;
            }

            return propertyTensor.arraySync()[0];
        }
        
        // Get the image dimensions (i.e. width, height, channels) based on the shape (of a tensor or model)
        function getImageDimensions(shape) {
            let height, width, channels, batchSize;

            switch (shape.length) {
                case 3:
                    [height, width, channels] = shape;
                    return { width, height, channels };
                case 4:
                    [batchSize, height, width, channels] = shape;
                    return { width, height, channels };
                default:
                    // E.g. a greyscale image will be 2D
                    throw new Error("The provided shape is not representing a rgb(w) image (because it is not 3D or 4D)");
            }
        }

        async function handleMsg(msg) {
            // The input message can contain a number of settings that overrule the node settings.
            // Note: don't simple use || because otherwise all 0 values in the msg will ignored...
            if (!msg.hasOwnProperty("minimumScore")) {
                msg.minimumScore = node.minimumScore;
            }
            if (!msg.hasOwnProperty("minimumHeight")) {
                msg.minimumHeight = node.minimumHeight;
            }
            if (!msg.hasOwnProperty("maximumHeight")) {
                msg.maximumHeight = node.maximumHeight;
            }
            if (!msg.hasOwnProperty("minimumWidth")) {
                msg.minimumWidth = node.minimumWidth;
            }
            if (!msg.hasOwnProperty("maximumWidth")) {
                msg.maximumWidth = node.maximumWidth;
            }
            if (!msg.hasOwnProperty("maximumDetections")) {
                msg.maximumDetections = node.maximumDetections;
            }
            if (!msg.hasOwnProperty("minimumAspectRatio")) {
                msg.minimumAspectRatio = node.minimumAspectRatio;
            }
            if (!msg.hasOwnProperty("maximumAspectRatio")) {
                msg.maximumAspectRatio = node.maximumAspectRatio;
            }

            // Start a new execution context for TensorFlow.js, to manage the lifecycle of tensors and other resources.
            // When a new scope is started, any tensors that are created within that scope will be tracked until the scope is ended.
            // Note that we use this instead of tf.tidy, because we had memory leaks when putting the below code in an async function.
            tf.engine().startScope();

            let imageTensor, originalImageTensor, detectionResult, inputImage;            

            try {
                msg.executionTimes = {};

                // ---------------------------------------------------------------------------------------------------------
                // 1. Decode the input image (to raw pixels) if it is type 'jpeg'
                // ---------------------------------------------------------------------------------------------------------

                switch (node.inputFormat) {
                    case "jpg":
                        inputImage = msg.payload;

                        if (!Buffer.isBuffer(inputImage) || inputImage[0] !== 0xFF || inputImage[1] !== 0xD8 || inputImage[2] !== 0xFF) {
                            node.error("The msg.payload doesn't contain a jpeg image");
                            return;
                        }

                        // Decode the image and convert it to a tensor (in this case it will become 3D tensors based on the encoded bytes).
                        // We could also use tfjs decodeImage function, that supports BMP, GIF, JPEG and PNG.  
                        // But jpeg will be mostly used for ip camera's, and this makes it faster to output jpeg images in case of 'passthrough'.
                        imageTensor = tf.node.decodeJpeg(inputImage);
                        break;
                    case "raw":
                        inputImage = msg.payload.data;

                        // Create an image tensor for the raw input image buffer
                        imageTensor = tf.tensor3d(inputImage, [msg.payload.height, msg.payload.width, msg.payload.channels], 'int32');
                }

                // Keep a reference to the original image tensor, because after the recognition part we will need this again
                originalImageTensor = imageTensor;

                msg.executionTimes.decoding = getDuration();

                // ---------------------------------------------------------------------------------------------------------
                // 2. Prepare the input image for the selected model
                // ---------------------------------------------------------------------------------------------------------

                // A color image can be represented by different shapes of sensors:
                //   - A 3D tensor (HWC): shape = [ image height, image width, number of color channels ] where color channels are e.g. RGB
                //   - A 4D tensor (NHWC): shape = [ batch size, image height, image width, number of color channels ] where batch is used for multiples images e.g. for training
                //            --> for object detection the batch size will always be 1
                // If the image is a 3D tensor but the model requires a 4D sensor, then add the batch size as first dimension.
                // Normally the batch size is used to train the model with a batch of N images.
                // However for object detection there will only be 1 image involved, so the batch size will need to be 1.
                // Otherwise the prediction will throw an exception: "Input tensor shape mismatch: expect '1,300,300,3', got '300,300,3'"
                if (node.tensorflowConfigNode.runtime !== "tfjs") {
                    if (imageTensor.shape.length == 3 && node.modelInputMetadata.shape.length == 4) {
                        let expandedImageTensor = tf.expandDims(imageTensor, 0);
                        imageTensor = expandedImageTensor;
                    }
                }

                // Get the dimensions of the input image
                let inputImageDimensions = getImageDimensions(imageTensor.shape);

                // Get the image size expected by the model, which corresponds to the size of the images that have been used to train the model.
                // These dimensions are stored in the model shape tensor, which contains the following information: [batchsize, height, width, colorCount].
                let requiredModelDimensions = getImageDimensions(node.modelInputMetadata.shape);

                if (requiredModelDimensions.height == -1 && requiredModelDimensions.width == -1) {
                    // A fully convolutional model processes the input image using a series of convolutional layers, that can work with images of any size.
                    // For example the Tensorflow Coco SSD model doesn't need to know the exact width and height of the input image, so it requires width and height -1.
                    // Instead, the model learns to detect objects based on patterns in the feature maps, which are independent of the input size.
                    // Therefore we skip the resizing...
                }
                else {
                    // Resize the input image if it's dimensions don't match with the model requirement
                    if (inputImageDimensions.height != requiredModelDimensions.height || inputImageDimensions.width != requiredModelDimensions.width) {
                        let resizedImageTensor;

                        // Otherwise the prediction will throw an exception.  For example: Input tensor shape mismatch: expect '1,300,300,3', got '1,480,640,3'.
                        switch(node.tensorflowConfigNode.resizeAlgorithm) {
                            case "bilinear":
                                // The 'bilinear' algorithm offers smooth transitions.
                                resizedImageTensor = tf.image.resizeBilinear(imageTensor, [requiredModelDimensions.width, modelHeight]);
                                imageTensor = resizedImageTensor;
                                break;
                            case "neirest":
                                // The 'neirest neighbour' algorithm is the fastest one (which still delivers enough accuracy).
                                resizedImageTensor = tf.image.resizeNearestNeighbor(imageTensor, [requiredModelDimensions.height, requiredModelDimensions.width]);
                                imageTensor = resizedImageTensor;
                                break;
                        }
                    }

                    if (node.modelInputMetadata.dtype === 'float32' && imageTensor.dtype === 'int32') {
                        // When the model requires dtype int32, all tensor elements contain RGB colors in the range from 0 to 255.
                        // When the model requires dtype int32, all tensor elements need to be normalized (to the range from 0 to 1).
                        // To accomplish that, divide all tensor elements by 255
                        let normalizedImageTensor = imageTensor.div(255.0);
                        imageTensor = normalizedImageTensor;
                    }

                    msg.executionTimes.resizing = getDuration();
                }

                // ---------------------------------------------------------------------------------------------------------
                // 3. Execute object detection on the image
                // ---------------------------------------------------------------------------------------------------------

                switch(node.tensorflowConfigNode.runtime) {
                    case "tfjs":
                        // The tfjs models are pretty easy to use, since all Tensor handling is done inside the tfjs models.
                        // Which means the input is an image, and the output is bboxes/scores/...
                        // This in contradiction to the other models (TfLite, Graph, ...), where all input/output processing needs to be done by this node.
                        // TODO don't use await to avoid uncaught exceptions
                        detectionResult = await node.model.detect(imageTensor, msg.maximumDetections);
                        break;
                    case "tflite":
                        // It is NOT required to normalize the input images (i.e. from values in the range [0, 255] to [0,1]).
                        // Because this model (which is a TfLite model converted for Coral TPU's) requires uint8 values, in contradiction to its original models.
                        // The model throws an exception if we normalize the input images: "Data type mismatch: input tensor expects 'uint8', got 'float32'"
                        // TODO catch errors instead of letting node-red crash
                        detectionResult = node.model.predict(imageTensor);
                        break;
                    case "graph":
                        // TODO can predict also be used here instead of executeAsync?
                        await node.model.executeAsync(imageTensor).then(result => {
                            detectionResult = result
                        }).catch(err => {
                            throw err;
                        });
                        break;
                }

                msg.executionTimes.object_detection = getDuration();
                
                // ---------------------------------------------------------------------------------------------------------
                // 4. Parse the object detection output (tensors)
                // ---------------------------------------------------------------------------------------------------------

                msg.classes = {};

                // Postprocessing is NOT required for runtime type 'tfjs', because the tfjs model has the postprocessing code inside the tfjs model
                if (node.tensorflowConfigNode.runtime === "tfjs") {
                    msg.payload = detectionResult;
                }
                else {
                    msg.payload = [];

                    // Get the required information from the specified output tensors
                    let bboxes      = getOutputProperty(detectionResult, "bboxes"      , node.tensorflowConfigNode.bboxProperty , node.tensorflowConfigNode.bboxPropertyType);
                    let classes     = getOutputProperty(detectionResult, "classes"     , node.tensorflowConfigNode.classProperty, node.tensorflowConfigNode.classPropertyType);
                    let scores      = getOutputProperty(detectionResult, "scores"      , node.tensorflowConfigNode.scoreProperty, node.tensorflowConfigNode.scorePropertyType);
                    let objectCount = getOutputProperty(detectionResult, "object count", node.tensorflowConfigNode.countProperty, node.tensorflowConfigNode.countPropertyType);

                    // Convert the 3 arrays to a single array (containing 3 properties), similar to the tfjs models
                    for(let i = 0; i < objectCount; i++) {
                        // The normalized bbox has the format [ymin, xmin, ymax, xmax]
                        let normalizedBbox = bboxes[i];
                        let x1, y1, x2, y2;

                        // Determine the bbox upper-left (x1,y1) and lower-right corner (x2,y2) points, based on the specified model output bbox format
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
                        x1 *= inputImageDimensions.width;
                        x2 *= inputImageDimensions.width;
                        y1 *= inputImageDimensions.height;
                        y2 *= inputImageDimensions.height;

                        // The detection result can contain coordinates outside of the image dimensions, so force them to be withing the image (via min and max).
                        x1 = Math.max(0, x1);
                        y1 = Math.max(0, y1);
                        x2 = Math.min(inputImageDimensions.width, x2);
                        y2 = Math.min(inputImageDimensions.height, y2);

                        let width = Math.abs(x2 - x1);
                        let height = Math.abs(y2 - y1);

                        // For all runtime types we will send the bbox in the format [x_min, y_min, width, height]
                        let denormalizedBbox = [x1, y1, width, height];
            
                        // The info for 1 detected object is spread across 3 separate arrays
                        msg.payload.push({
                            bbox: denormalizedBbox,
                            classIndex: classes[i],
                            class: node.labels[classes[i]],
                            score: scores[i]
                        });
                    }
                }

                // Store the bbox dimensions in the output
                msg.payload.forEach(function(detectedObject, index) {
                    let width = detectedObject.bbox[2];
                    let height = detectedObject.bbox[3];

                    detectedObject.size = {
                        width: width,
                        height: height,
                        aspectRatio: width / height
                    }
                })

                // ---------------------------------------------------------------------------------------------------------
                // 5. Filter out unwanted detected objects
                // ---------------------------------------------------------------------------------------------------------

                // Sort the filtered array of detected objects (by descending score).
                // Because that is easier for next nodes in the flow to get the top-N detections.
                // And it also makes sure we keep below the detections with the highest scores, in case a maximum number of detections has been specified.
                msg.payload.sort((a, b) => (a.score < b.score) ? 1 : -1);

                // When a maximum number of detections has been specified, then filter out all other detections that we don't need
                if (msg.maximumDetections < msg.payload.length) {
                    msg.payload = msg.payload.slice(0, msg.maximumDetections);
                }

                // Only keep detected objects whose score is at least equal to the minimum score
                msg.payload = msg.payload.filter(function (detectedObject) {
                    return detectedObject.score >= msg.minimumScore;
                });

                // Only keep detected objects whose class is in the class filter array (only if there is a non-empty class filter array)
                msg.payload = msg.payload.filter(function (detectedObject) {
                    return node.classFilter.length == 0 || node.classFilter.includes(detectedObject.class);
                });

                // Only keep detected objects whose bounding box that has the minimum required size
                msg.payload = msg.payload.filter(function (detectedObject) {
                    let width = detectedObject.size.width;
                    let height = detectedObject.size.height;
                    let aspectRatio = detectedObject.size.aspectRatio;

                    return width >= msg.minimumWidth &&
                           width <= msg.maximumWidth && 
                           height >= msg.minimumHeight &&
                           height <= msg.maximumHeight &&
                           aspectRatio >= msg.minimumAspectRatio &&
                           aspectRatio <= msg.maximumAspectRatio
                });

                // Count the number of (filtered) detections for each detected class (i.e. class statistics)
                for (let j=0; j < msg.payload.length; j++) {
                    if (msg.payload[j] && msg.payload[j].hasOwnProperty("class")) {
                        msg.classes[msg.payload[j].class] = (msg.classes[msg.payload[j].class] || 0 ) + 1;
                    }
                }

                // Total number of detected objecdts in the output message
                msg.detectionCount = msg.payload.length;

                msg.executionTimes.parse_detection_result = getDuration();
                
                // At this point the image tensor has been manipulated to fullfill the requirements of the specified model.
                // However from this point on, we don't need that manipulated tensor anymore.
                // We will continue now with the original unmodified tensor, containing the original input image.
                imageTensor = originalImageTensor;

                // ---------------------------------------------------------------------------------------------------------
                // 6. Draw the bounding boxes on top of the original non-resized raw image
                // ---------------------------------------------------------------------------------------------------------

                // Only execute the bbox drawing code when objects have been detected, to avoid converting images (rgb -> rgba -> rgb) when no bboxes will be drawn
                if (node.passthru === "bbox" && msg.payload.length > 0) {
                    switch(inputImageDimensions.channels) {
                        case 3:
                            // When the image tensor contains RGB (i.e. 3 channels), a fourth alpha channel needs to be added.
                            // PureImage requires RGBA data to display inside it's canvas, so no RGB data is allowed.
                            // This is similar to the html canvas (see https://developer.mozilla.org/en-US/docs/Web/API/ImageData/data).
                            // So add an alpha channel to the image if it is missing.  This will be the case for decoded jpeg's, since jpeg has no transparency support.
                            // We have asked the Tensorflow team to add alpha channel support for jpeg (see https://github.com/tensorflow/tfjs/issues/6911).
                            let alphaTensorShape, axis;

                            switch (imageTensor.shape.length) {
                                case 3:
                                    axis = 2;
                                    alphaTensorShape = [inputImageDimensions.height, inputImageDimensions.width, 1];
                                    break;
                                case 4:
                                    axis = 3;
                                    alphaTensorShape = [1, inputImageDimensions.height, inputImageDimensions.width, 1];
                                    break;
                            }

                            // Set all the values in the alpha channel tensor to 1, which means that all pixels are fully opaque.
                            // Make sure the datatype (int32, float32) of the alpha channel tensor is equal to the dtype of the image tensor.
                            let alphaChannelTensor = tf.fill(alphaTensorShape, 1, imageTensor.dtype);

                            // Add the alpha channel to the other channels, i.e. at the channel axis (= index of the channels in the shape array)
                            let rgbaImageTensor = tf.concat([imageTensor, alphaChannelTensor], axis);
                            imageTensor = rgbaImageTensor;

                            msg.executionTimes.rgb_to_rgba = getDuration();
                            break;
                        case 4:
                            // Seems that we are already dealing with an rgbw tensor, so no conversion of the image tensor needed
                            break;
                        default:
                            throw new Error("The provided shape is not representing a rgb(w) image (because it has not 3 or 4 channels)");
                    }

                    // Get a `TypedArray` view of the tensor's data buffer.
                    // Don't use dataSync because it creates a new array containing a copy of the tensor's data.
                    let dataView = await imageTensor.data();
                    
                    let rawImage;

                    // Access the raw image data from the `TypedArray` view via an array.
                    // Since the TypedArray view is backed by the same data buffer as the tensor, no cloning is necessary.
                    switch (imageTensor.dtype) {
                        case "float32":
                            rawImage = new Float32Array(dataView.buffer);
                            break;
                        case "int32":
                            rawImage = new Int32Array(dataView.buffer);
                            break;
                        default:
                            throw new Error("Cannot get raw image from tensor with dtype " + imageTensor.dtype);
                    }

                    // Create a new empty PureImage 2D canvas
                    let pimg = pureimage.make(inputImageDimensions.width, inputImageDimensions.height);
                    let ctx = pimg.getContext('2d');

                    // Some stuff we draw on top of the image might need to be scaled, relative to the image size.
                    // This way we can ensure that the stuff that we draw remains visible and proportional regardless of the image dimensions.
                    let scale = parseInt((inputImageDimensions.width + inputImageDimensions.height) / 500 + 0.5);

                    // Store the original non-resized raw image (which has been set aside already in the beginning) inside the PureImage 2D canvas
                    ctx.bitmap.data = new Uint8ClampedArray(rawImage);

                    let textHeight = scale * 8;
   
                    ctx.lineJoin = 'bevel'; // TODO why is this needed?

                    // Draw all the bounding boxes in the PureImage 2D canvas
                    msg.payload.forEach(function (detectedObject, index) {
                        let bboxColor, text;

                        if (node.colors && Array.isArray(node.colors) && node.colors.length > 0) {
                            // Lookup the index of the class to cross ref into color table
                            bboxColor = node.colors[detectedObject.classIndex];
                        }
                        else {
                            // Apply the specified default color
                            bboxColor = node.lineColor;
                        }

                        let x1 = detectedObject.bbox[0];
                        let y1 = detectedObject.bbox[1];
                        let width = detectedObject.bbox[2];
                        let height = detectedObject.bbox[3];

                        // Calculate the padding on all size, because there might not be enough room for the specified fixed padding
                        let paddingLeft = Math.min(node.bboxPadding, x1);
                        let paddingTop = Math.min(node.bboxPadding, y1);
                        let paddingRight = Math.min(node.bboxPadding, inputImageDimensions.width - x1 - width);
                        let paddingBottom = Math.min(node.bboxPadding, inputImageDimensions.height - y1 - height);

                        // Correct the coordinates of the bbox upper left (x1,y1) and lower right (x2,y2), to take into account the specified padding
                        x1 = detectedObject.bbox[0] - paddingLeft;
                        y1 = detectedObject.bbox[1] - paddingTop;
                        width = detectedObject.bbox[2] + paddingLeft + paddingRight;
                        height = detectedObject.bbox[3] + paddingTop + paddingBottom;

                        // The padding can result in a bbox partly outside the image dimensions, so force it to be within the image (via min and max).
                        x1 = Math.max(0, x1);
                        y1 = Math.max(0, y1);
                        width = Math.min(inputImageDimensions.width, width);
                        height = Math.min(inputImageDimensions.height, height);

                        if (node.annotationExpression && node.annotationExpression.trim() !== "") {
                            let annotationExpression = node.annotationExpression;
                            // Resolve all the variables in the annotation expression
                            annotationExpression = annotationExpression.replaceAll("{{class}}", detectedObject.class);
                            annotationExpression = annotationExpression.replaceAll("{{score}}", detectedObject.score.toFixed(2)); // Score rounded to 2 decimals
                            annotationExpression = annotationExpression.replaceAll("{{scorePercentage}}", Math.round(detectedObject.score * 100)); // Score rounded to 2 decimals
                            annotationExpression = annotationExpression.replaceAll("{{width}}", Math.round(detectedObject.size.width));
                            annotationExpression = annotationExpression.replaceAll("{{height}}", Math.round(detectedObject.size.height));
                            annotationExpression = annotationExpression.replaceAll("{{aspectRatio}}", detectedObject.size.aspectRatio.toFixed(2));
                            annotationExpression = annotationExpression.replaceAll("{{index}}", index);

                            // Support multiline text
                            // TODO check where the escaping (i.e. the second slash) is coming from
                            let texts = annotationExpression.split('\\n');

                            // Calculate the maximum width of all available text lines (and at least as wide as the bounding box.
                            // However the text can be wider than the bounding box.
                            let totalTextWidth = width;
                            texts.forEach(function (text, index) {
                                let textWidth = ctx.measureText(text).width;
                                totalTextWidth = Math.max(textWidth, totalTextWidth);
                            })

                            // In case of multiline annotation text, the text height depends on the number of text lines
                            let totalTextHeight = textHeight * texts.length;
 
                            // Calculate the rectangle (i.e. bbox) of the annotation label
                            let rect_x = x1 + width/2 - totalTextWidth/2;
                            let rect_y;
                            if (y1 >= totalTextHeight) {
                                // By default the annotation text is displayed above the bounding box.
                                rect_y = y1 - totalTextHeight;
                            }
                            else {
                                // When there is not enough space below the bounding box, then show the text above the bounding box
                                rect_y = y1 + height;
                            }

                            // Draw a filled rectangle below the text (if required)
                            if (node.annotationBackground == "rect") {
                                ctx.fillStyle = bboxColor;
                                ctx.fillRect(rect_x, rect_y, totalTextWidth, totalTextHeight);
                            }

                            // Draw the texts above the bounding box
                            texts.forEach(function (text, index) {
                                // The text should be centered horizontall (i.e. in the middle of the bounding box)
                                let text_x = x1 + width/2;
                                let text_y = rect_y - 4 + textHeight*index;

                                // The annotation text should be white if there is a filled rectangle below it
                                if (node.annotationBackground == "rect") {
                                    ctx.fillStyle = "white";
                                }
                                else {
                                    ctx.fillStyle = bboxColor;
                                }

                                // Draw the text filled with the same color as the bbox outline
                                ctx.font = textHeight + "pt 'Source Sans Pro'";
                                ctx.textAlign = "center";
                                ctx.textBaseline = "top";
                                ctx.fillText(text, text_x, text_y);
                            })
                        }

                        // Draw the bounding box (corrected with padding)
                        ctx.strokeStyle = bboxColor;
                        ctx.lineWidth = scale;
                        ctx.strokeRect(x1, y1, width, height);
                    });

                    // Convert the annotated image (from the PureImage canvas) - containing the bounding boxex - to a tensor
                    let annotatedImageTensor = tf.tensor3d(pimg.data, [inputImageDimensions.height, inputImageDimensions.width, 4], 'int32');
                    imageTensor = annotatedImageTensor;

                    msg.executionTimes.bbox_drawing = getDuration();
                }

                // ---------------------------------------------------------------------------------------------------------
                // 7. Add the image to the output message in the specified format
                // ---------------------------------------------------------------------------------------------------------

                if (node.passthru === "orig") {
                    // Pass the original input image to the output
                    msg.image = {
                        data: inputImage,
                        type: node.inputFormat
                    }

                    // TODO use https://www.npmjs.com/package/image-type to determine msg.image.type
                    // Surely we only support jpeg ? as the actual image type ? (or rather tfjs does ?)
                }
                else {
                    // Get the dimensions of the input image again (because the number of channels might have been changed by bbox drawing)
                    inputImageDimensions = getImageDimensions(imageTensor.shape);

                    if (node.outputFormat == "jpg") { // TODO the output image format (jpeg, raw, ...) should be adjustable in the config screen
                        switch(inputImageDimensions.channels) {
                            case 3:
                                // Seems that we are already dealing with an rgb tensor, so no conversion of the image tensor needed
                                break;
                            case 4:
                                // The encodeJpeg does not support an alpha channel, so convert the annotated image tensor from RGBA to RGB
                                let [channel_r, channel_g, channel_b, channel_a] = tf.split(imageTensor, 4, 2); // TODO check whether this works for both 3D and 4D tensors
                                let rgbImageTensor = tf.concat([channel_r, channel_g, channel_b], 2);
                                 tf.dispose([channel_r, channel_g,channel_b, channel_a]); // TODO remove this??
                                imageTensor = rgbImageTensor;

                                msg.executionTimes.rgba_to_rgb = getDuration();

                                break;
                            default:
                                throw new Error("The provided shape is not representing a rgb(w) image (because it has not 3 or 4 channels)");
                        }

                        // Encode the RGB tensor to a JPEG image
                        let jpegData = await tf.node.encodeJpeg(imageTensor, 'rgb');

                        // Convert the jpeg image from Uint8Array to a buffer
                        msg.image = {
                            data: Buffer.from(jpegData),
                            type: "jpg"
                        }

                        msg.executionTimes.encoding = getDuration();
                    }
                    else {
                        // TODO test whether we should use dataSync i.e. a clone of the data before sending it to the next nodes in the flow
                        let imageArray = await imageTensor.data();

                        // Convert the raw image from Uint32Array to a buffer
                        msg.image = {
                            data: Buffer.from(imageArray),
                            type: "raw" // TODO: real mime type?
                        }
                    }
                }

                msg.image.width = inputImageDimensions.width;
                msg.image.height = inputImageDimensions.height;
                msg.image.channels = inputImageDimensions.channels;

                // ---------------------------------------------------------------------------------------------------------
                // 8. Statistics
                // ---------------------------------------------------------------------------------------------------------

                // Get information about the memory usage 
                let usedMemory = process.memoryUsage();
                msg.memory_stats = {
                    rss: Math.round((usedMemory.rss / 1024 / 1024) * 100) / 100,
                    heapTotal: Math.round((usedMemory.heapTotal / 1024 / 1024) * 100) / 100,
                    heapUsed: Math.round((usedMemory.heapUsed / 1024 / 1024) * 100) / 100,
                    external: Math.round((usedMemory.external / 1024 / 1024) * 100) / 100,
                    arrayBuffers: Math.round((usedMemory.arrayBuffers / 1024 / 1024) * 100) / 100
                }

                let tfjsMemory = tf.memory();
                msg.tfjs_stats = {
                    numTensors: tfjsMemory.numTensors, // tensor count before disposal
                    numDataBuffers: tfjsMemory.numDataBuffers,
                    numBytes: tfjsMemory.numBytes
                }

                msg.node_stats = {
                    skippedMsgCount: node.skippedMsgCount,
                    skippedMsgPercentage: 100 * node.skippedMsgCount / (node.skippedMsgCount + 1)
                }

                // Start counting the skipped messages again
                node.skippedMsgCount = 0;

                // Calculate the total execution time
                msg.busyTime = 0;
                Object.values(msg.executionTimes).forEach(function (executionTime, index) {
                    msg.busyTime += executionTime;
                });
                
                msg.busyPercentage = (msg.busyTime / (msg.busyTime + msg.idleTime)) * 100;
                msg.busyPercentage = +msg.busyPercentage.toFixed(2);

                node.send(msg);

                // The 'idle' time calculation starts from the moment the current message has been processed completely.
                node.startTime = process.hrtime();
            }
            catch (error) {
                node.error(error, msg);
                console.log(error); // Print stacktrace
            }

            // Mark all variables - created within the current execution context - for disposal, so that Tensorflow.js can decrement their reference
            // counts by one. If the reference count for any of these tensors drops to zero, the TensorFlow.js engine's garbage collector will 
            // automatically dispose of them to free up memory.  These variables will be tensor data and any additional metadata associated with this.
            // When you create a new tensor, the TensorFlow.js engine increments the reference count for that tensor by one. When you dispose of a 
            // tensor, the reference count for that tensor is decremented by one. When the reference count for a tensor reaches zero, meaning that 
            // no variables or other objects are referencing that tensor, the TensorFlow.js engine's garbage collector will automatically dispose 
            // of the tensor to free up memory.
            tf.disposeVariables();

            // Let TensorFlow.js know that the current execution context has ended, and that any resources associated with that context 
            // can be released.  TensorFlow.js memory is allocated and managed by the underlying WebGL context, which is implemented by a Node.js
            // package called headless-gl.  When tensors are disposed, TensorFlow.js releases the memory associated with that tensor back to the 
            // WebGL context, allowing it to be reused for other operations. However, the WebGL context doesn't necessarily release this memory 
            // immediately, as it may keep the memory allocated in case it's needed for future operations...
            tf.engine().endScope();

            // Allow new messages after the async function call has completely ended
            node.busy = false;
        }

        node.on('input', function (msg) {
            // Allow replacing the labels array dynamically - either as an array or a csv string
            // If any labels are missing from the array then don't report/draw boxes later
            // If any colors are missing from the array then just use the default (magenta) later
            if (Array.isArray(msg.labels)) { node.labels = msg.labels; }
            if (typeof msg.labels === "string") { node.labels = msg.labels.split(','); }
            if (Array.isArray(msg.colors)) { node.colors = msg.colors; }
            if (typeof msg.colors === "string") { node.colors = msg.colors.split(','); }

            if (typeof msg.payload === "string") {
                if (msg.payload.startsWith("data:image/jpeg")) {
                    msg.payload = Buffer.from(msg.payload.split(";base64,")[1], 'base64');
                }
                else { msg.payload = fs.readFileSync(msg.payload); }
            }

            if (!msg.payload) {
                return;
            }

            if (!node.model) {
                node.warn("No object detection because the model is not loaded yet");
                return;
            }

            // Since all code in this node is being called synchronously (using await statements), this node will execute all processing steps
            // on image N before it will start processing the next image N+1.  Since it is not possible this way to process multiple images with
            // a single node in parallel, it is also useless to send image N+1 while the previous image N is still being processed.  That would
            // be no problem for a few images, but e.g. for an endless mjpeg stream the RSS memory will run full very quickly by queued calls.
            if (node.busy) {
                console.log("No object detection because node still busy with previous image");
                node.skippedMsgCount++;
                return;
            }

            // Immediately set the node as busy before the async function is called, to avoid that multiple async functions are started in parallel
            node.busy = true;

            // The idle time is the time since the end of the previous message processing.
            msg.idleTime = getDuration();

            // Do all the image processing in an asynchronous call, to make sure that the "input" event handler code of node is able to accept 
            // new messages (which might be skipped when the node is busy).  Use setTimeout because even a simple async call via 'await' does not
            // allow the event loop to continue processing messages while the handleMsg function is being executed.  When the events get queued
            // in NodeJs, all the queued messages will cause memory to run full very fast. 
            setTimeout(function() {
                handleMsg(msg);
            }, 0);
        });

        node.on("close", function () {
            node.status({});
            node.model.dispose();
            node.model = null;
            node.fnt = null;
            node.busy = false;
        });
    }
    RED.nodes.registerType("tensorflowObjDet", TensorFlowObjDet);
};
