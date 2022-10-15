 module.exports = function(RED) {
    var settings = RED.settings;

    function TensorFlowObjDetConfig(config) {
        RED.nodes.createNode(this,config);
        this.runtime           = config.runtime;
        this.model             = config.model;
        this.modelType         = config.modelType;
        this.labels            = config.labels;
        this.labelsType        = config.labelsType;
        this.colors            = config.colors;
        this.colorsType        = config.colorsType;
        this.bboxProperty      = config.bboxProperty;
        this.bboxPropertyType  = config.bboxPropertyType;
        this.classProperty     = config.classProperty;
        this.classPropertyType = config.classPropertyType;
        this.scoreProperty     = config.scoreProperty;
        this.scorePropertyType = config.scorePropertyType;
        this.countProperty     = config.countProperty;
        this.countPropertyType = config.countPropertyType;
        this.bboxFormat        = config.bboxFormat;
        this.resizeAlgorithm   = config.resizeAlgorithm;
    }

    RED.nodes.registerType("tensorflowObjDetConfig", TensorFlowObjDetConfig);
}