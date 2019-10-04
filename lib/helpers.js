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

/**
 * Squared error function
 * @param {Number} target target / ideal output
 * @param {Number} output actual / real output
 * @returns {Number}
 */
export function squaredError(target, output) {
    return 0.5 * (target - output) ** 2;
}

/**
 * Shuffle an array using the Fisher-Yates algorithm
 * @param {Array} array input array
 * @return {Array}
 */
// see: https://stackoverflow.com/questions/6274339/how-can-i-shuffle-an-array
export function shuffle(array) {
    let counter = array.length;

    while (counter > 0) {
        let index = Math.floor(Math.random() * counter);
        counter--;

        let temp = array[counter];
        array[counter] = array[index];
        array[index] = temp;
    }

    return array;
}
