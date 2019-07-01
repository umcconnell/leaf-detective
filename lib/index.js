/** @module leaf-detective/main */

import { plus } from "./helpers.js";

/**
 * Class representing layer weights
 * @extends Array
 */
export class Weights extends Array {
    /**
     * Creates weights matrix
     * @param {number} width width of weights matrix
     * @param {number} height height of weights matrix
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
        return this.map(row => row.map(el => Math.random()));
    }

    /**
     * Fills given data into matrix
     * @param {array} data array of data to populate weights matrix with; must
     * be of same width and height as matrix
     * @returns {Weights}
     */
    populate(data) {
        return this.map((row, i) => row.map((el, j) => data[i][j]));
    }
}
