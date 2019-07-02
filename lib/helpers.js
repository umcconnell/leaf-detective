/** @module leaf-detective/helpers */

/**
 * Add two numbers
 * @param {(Number|String)} a first number
 * @param {(Number|String)} b second number
 * @example
 * plus(1, 2)
 *
 * //=> 3
 * @returns {Number} sum of both numbers
 */
export function plus(a, b) {
    return a + b;
}

/**
 * Map an input number to an output number using the sigmoid function
 * @param {Number} x input
 * @example
 * sigmoid(0)
 *
 * //=> 0.5
 * @returns {Number} output
 */
export function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}
