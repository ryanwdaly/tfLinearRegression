const height = 400;
const width = 400;

const learningRate = 0.2;
const optimizer = tf.train.sgd(learningRate);

let xs = [];
let ys = [];

let m, b;

function setup() {
    createCanvas(width, height);
    background(0);

    m = tf.variable(tf.scalar(random(1)));
    b = tf.variable(tf.scalar(random(1)));
}



function draw() {
    background(0);
    
    stroke(255);
    strokeWeight(8);
    for (let i = 0; i < xs.length; i++) {
        let px = map(xs[i], 0, 1, 0, width);
        let py = map(ys[i], 0, 1, 0, height);
        point(px, py);
        
    }
}

//helpers
function mousePressed() {
    let x = map(mouseX, 0, width, 0, 1);
    let y = map(mouseY, 0, height, 0, 1);
    xs.push(x);
    ys.push(y);
}

function predict(x) {
    const xs = tf.tensor1d(x);
    const ys = xs.mul(m).add(b);

    return ys;
}

function loss(pred, labels) {
    return pred.sub(labels).squared().mean();
}