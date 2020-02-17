import { sigmoid as sig } from "./helpers.js";

export class Activation {
    constructor(source, derivative) {
        this.source = source;
        this.derivative = derivative;
    }
}

// See: https://en.wikipedia.org/wiki/Activation_function
export const arctan = new Activation(
    x => Math.atan(x),
    x => 1 / (x ** 2 + 1)
);

export const elliotsig = new Activation(
    x => x / (1 + Math.abs(x)),
    x => 1 / (1 + Math.abs(x)) ** 2
);

export const gaussian = new Activation(
    x => Math.exp(-(x ** 2)),
    x => -2 * x * Math.exp(-(x ** 2))
);

export const identity = new Activation(
    x => x,
    x => 1
);

export const relu = new Activation(
    x => Math.max(0, x),
    x => (x <= 0 ? 0 : 1)
);

export const sigmoid = new Activation(
    x => sig(x),
    x => sig(x) * (1 - sig(x))
);

export const sinc = new Activation(
    x => (x == 0 ? 1 : Math.sin(x) / x),
    x => (x == 0 ? 0 : Math.cos(x) / x - Math.sin(x) / x ** 2)
);

export const sinusoid = new Activation(
    x => Math.sin(x),
    x => Math.cos(x)
);

export const softplus = new Activation(
    x => Math.log(1 + Math.exp(x)),
    x => sig(x)
);
