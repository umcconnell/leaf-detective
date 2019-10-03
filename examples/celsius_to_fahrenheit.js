let { Network } = require("../lib/index.js");
let { plus } = require("../lib/helpers.js");

console.log(`Celsius to Fahrenheit conversion:
################
The neural network consists of 2 hidden layers with the size 4 and 2.
It is given a temperature between 1 and 100 in celsius and expected to return
the corresponding temperature in degrees fahrenheit.
################
`);

let t0 = new Date().getTime();

// 1. Create network
let network = new Network([1, 4, 2, 1])
    .connect()
    .addWeights()
    .addBiases();

// 2. Generate Data
let data = new Array(1000).fill(0).map(_ => {
    let celsius = Math.floor(Math.random() * (100 - 0 + 1)) + 0;

    //  Normalize temperature by dividing by 212
    // (highest possible temperature = 100°C = 212°F)
    return {
        input: [celsius / 212],
        expected: (celsius * 1.8 + 32) / 212
    };
});

const testData = data.splice(800, 200);

// 3. Train network
console.log("Training neural network...");

data.forEach(train => {
    network.populate(train.input);
    for (let i = 0; i < 1000; i++) {
        network.run().backpropagate([train.expected], 0.01, 1);
    }
});

let t1 = new Date().getTime();

console.log("Done Training");
console.log(`Took ${t1 - t0} milliseconds`);
console.log("-------------");

// 4. Test network
console.log("Testing neural network...");
console.log(
    `Average error: ${testData
        .map(train => {
            network.populate(train.input).run();

            let actual = network[network.length - 1].neurons[0],
                diff = Math.abs(train.expected - actual) * 212;

            return diff;
        })
        .reduce(plus, 0) / testData.length}`
);
