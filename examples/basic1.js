let { Network } = require("../lib/index.js");
let { sigmoid, plus } = require("../lib/helpers.js");

console.log(`Basic Example 1:
################
The neural network consists of 2 hidden layers with the size 3.
It is given a list of 3 random bits, such as [1, 0, 1], and is
supposed to return the 1st value.
################
`);

// 1. Create network
let network = new Network([3, 3, 3, 1])
    .connect()
    .addWeights()
    .addBiases();

// 2. Generate Data
let data = new Array(200).fill(0).map(_ => {
    let input = new Array(3).fill(0).map(_ => Math.round(Math.random()));

    return {
        input,
        expected: input[0]
    };
});

const testData = new Array(50).fill(0).map(_ => {
    let input = new Array(3).fill(0).map(_ => Math.round(Math.random()));

    return {
        input,
        expected: input[0]
    };
});

// 3. Train network
console.log("Training neural network...");

data.forEach(train => {
    network.populate(train.input);
    for (let i = 0; i < 50; i++) {
        network.run(sigmoid).backpropagate([train.expected], 0.8, 1);
    }
});

console.log("Done Training");
console.log("-------------");

// 4. Test network
console.log("Testing neural network...");
console.log(
    `Average error: ${testData
        .map(train => {
            network.populate(train.input).run(sigmoid);

            let actual = network[network.length - 1].neurons[0],
                diff = Math.abs(train.expected - actual);

            return diff;
        })
        .reduce(plus, 0) / testData.length}`
);