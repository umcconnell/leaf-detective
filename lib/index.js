/** @module leaf-detective/main */

import { plus } from "./helpers.js";

/**
 * Class representing layer weights
 * @extends Array
 */
export class Weights extends Array {
    /**
     * Creates weights matrix
     * @param {Number} width width of weights matrix
     * @param {Number} height height of weights matrix
     */
    constructor(width, height) {
        super(height)
            .fill(0)
            .forEach((_, i, arr) => (arr[i] = new Array(width).fill(0)));
    }

    /**
     * Fills weights matrix with random numbers between 0 (included) and 1
     * (excluded)
     * @returns {Weights}
     */
    fillRandom() {
        return this.map(row => row.map(_ => Math.random()));
    }

    /**
     * Fills given data into matrix
     * @param {Array} data array of data to populate weights matrix with; must
     * be of same width and height as matrix
     * @returns {Weights}
     */
    populate(data) {
        return this.map((row, i) => row.map((_, j) => data[i][j]));
    }
}

/**
 * Class representing layer biases
 * @extends Array
 */
export class Biases extends Array {
    /**
     * Creates a bias matrix
     * @param {Number} height height of bias matrix; must correspond to output
     * layers length
     */
    constructor(height) {
        super(height).fill(0);
    }

    /**
     * Fills bias matrix with random numbers between -1 (included) and 1
     * (excluded)
     * @returns {Biases}
     */
    fillRandom() {
        return this.map(_ => Math.random() * 2 - 1);
    }

    /**
     * Fills given data into matrix
     * @param {Array} data array of data to populate bias matrix with; must be
     * same width as matrix
     * @returns {Biases}
     */
    populate(data) {
        return this.map((_, i) => data[i]);
    }
}

/**
 * Class representing layer matrix
 * @example
 * // Import constructors and helpers
 * import { Layer, Weights, Biases } from "./index.js";
 * import { sigmoid } from "./helpers.js";
 *
 * // Create Layers
 * let inputLayer = new Layer(5);
 * let outputLayer = new Layer(2);
 *
 * // Create weights and biases
 * let weights = new Weights(inputLayer.length, outputLayer.length).fillRandom();
 * let biases = new Biases(outputLayer.length).populate([-10, 10]);
 *
 * // Connect weights and outputLayer to inputLayer
 * inputLayer.addWeights(weights).connect(outputLayer);
 * // Connect biases to outputLayer
 * outputLayer.addBiases(biases);
 *
 * // Add data and run neural net
 * inputLayer.populate([1, 1, 1, 1, 1]).run().apply(sigmoid);
 */
export class Layer {
    /**
     * Creates a layer with given amount of neurons
     * @param {Number} length length / amount of neurons of layer
     */
    constructor(length) {
        this.neurons = new Array(length).fill(0);
        this.length = length;

        this.weights = null;
        this.biases = null;

        this.next = null;
        this.previous = null;
    }

    /**
     * Adds Biases object to layer
     * @param {Biases} biases biases object from Biases constructor; belongs to ouput layer
     * @example
     * let inputLayer = new Layer(5);
     * let outputLayer = new Layer(2);
     *
     * outputLayer.addBiases(biases);
     * @returns layer
     */
    addBiases(biases) {
        if (!(biases instanceof Biases))
            throw new Error("Please pass a bias object");
        this.biases = biases;
        return this;
    }

    /**
     * Adds Weights object to layer
     * @param {Weights} weights weights object from Weights constructor; belongs to input layer
     * @example
     * let inputLayer = new Layer(5);
     * let outputLayer = new Layer(2);
     *
     * inputLayer.addWeights(weights);
     * @returns layer
     */
    addWeights(weights) {
        if (!(weights instanceof Weights))
            throw new Error("Please pass a weights object");
        else if (!weights.every(row => row.length === this.length))
            throw new Error(
                "Width of weight matrix must be equal to amount of neurons"
            );
        this.weights = weights;
        return this;
    }

    /**
     * Applys given function to all neurons of the layer
     * @param {Function} func mapping function called with neuron value and index
     * @example
     * outputLayer.apply(sigmoid);
     * @returns layer
     */
    apply(func) {
        this.neurons = this.neurons.map((neuron, i) => func(neuron, i));
        return this;
    }

    /**
     * Connects an input layer with an output layer
     * @param {Layer} layer output layer object
     * @example
     * let inputLayer = new Layer(5);
     * let outputLayer = new Layer(2);
     *
     * inputLayer.connect(outputLayer)
     * @returns layer
     */
    connect(layer) {
        if (!(layer instanceof Layer)) throw new Error("Please pass a layer");
        this.next = layer;
        layer.previous = this;
        return this;
    }

    /**
     * Populates neurons with data
     * @param {Array} data array of data to populate layer / neurons with; must be same length as layer
     * @returns layer
     */
    populate(data) {
        if (data.length !== this.length)
            throw new Error(
                `Please pass data of length ${this.neurons.length}`
            );
        this.neurons = [...data];
        return this;
    }

    /**
     * Runs connection between two layers by applying weights to input layer's neurons and biases to resulting output layer's neurons
     * @example
     * let inputLayer = new Layer(5);
     * let outputLayer = new Layer(2);
     *
     * inputLayer.connect(outputLayer).run()
     * @returns connected output layer
     */
    run() {
        this.next.neurons = this.next.neurons
            .map((_, row) =>
                this.neurons
                    .map((neuron, col) => neuron * this.weights[row][col])
                    .reduce(plus, 0)
            )
            .map((output, row) => output + this.next.biases[row]);
        return this.next;
    }
}
