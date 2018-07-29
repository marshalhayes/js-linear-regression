let X = [];
let Y = [];

let m, b;

let reset = false;

let slider;
let optimizerSelect;
let learningRate = 0.3;

const f = (pred, label) => pred.sub(label).square().mean(); // mean squared error

function setup() {
  canvas = createCanvas(windowWidth, 400);
  canvas.parent('canvas');
  canvas.mouseClicked(() => {
    if (mouseButton === LEFT) {
      // Normalize x, y to (-1, 1) and add it to data
      let x = map(mouseX, 0, width, -1, 1);
      let y = map(mouseY, 0, height, 1, -1);
      X.push(x);
      Y.push(y);
    }
  });

  slider = createSlider(0.001, 1, 0.3, 0.001);
  slider.parent('canvas-wrapper');
  slider.style('width', '150px');
  slider.style('display', 'block');
  slider.input(resetCanvas); // When the slider is moved, reset the canvas

  optimizerSelect = createSelect();
  optimizerSelect.parent('canvas-wrapper');
  optimizerSelect.option('sgd');
  optimizerSelect.option('adam');
  optimizerSelect.changed(resetCanvas);

  let show = createButton('Show Data');
  show.parent('canvas-wrapper');
  show.mouseClicked(showData);

  let stop = createButton('Stop');
  stop.parent('canvas-wrapper');
  stop.mouseClicked(() => {
    noLoop();
  });

  let clear = createButton('Reset');
  clear.parent('canvas-wrapper');
  clear.mouseClicked(() => {
    resetCanvas();
  });

  resetCanvas();
}

function draw() {
  background(0);

  learningRate = slider.value();
  fill(255).strokeWeight(0).textSize(15);
  text(`learning rate = ${learningRate}`, 20, 20);
  text(`y = ${m.dataSync()}x + ${b.dataSync()}`, 20, 35);

  tf.tidy(() => {
    if (X.length > 0) {
      const y = tf.tensor1d(Y);
      optimizer.minimize(() => {
        loss = f(tf.tensor1d(X).mul(m).add(b), y);
        loss.data().then((mse) => {
          fill(255).strokeWeight(0).textSize(15);
          text(`mean squared error = ${mse}`, 20, 50);
        });
        return loss;
      });

      const lineX = [-1, 1];

      const ys = tf.tensor1d(lineX).mul(m).add(b);
      let lineY = ys.data().then((y) => {

        // reverse normalization of x, y
        let x1 = map(lineX[0], -1, 1, 0, width);
        let x2 = map(lineX[1], -1, 1, 0, width);
        let y1 = map(y[0], -1, 1, height, 0);
        let y2 = map(y[1], -1, 1, height, 0);

        stroke(139, 0, 139);
        strokeWeight(2);
        line(x1, y1, x2, y2);
      });
    }
  });

  strokeWeight(8)
  stroke(250);
  for (let i = 0; i < X.length; i++) {
    let px = map(X[i], -1, 1, 0, width);
    let py = map(Y[i], -1, 1, height, 0);
    point(px, py);
  }
}

function showData() {
  let data = document.getElementById('data');
  data.innerText = null;
  if (X.length > 0) {
    // Show x, y pairs
    for (let i = 0; i < X.length; i++) {
      data.innerText += `[${X[i]}, ${Y[i]}]\r\n`;
    }
  }
}

/* resetCanvas is causing a memory leak...Each time resetCanvas is called
more and more tensors are created. A fix is coming soon.

It is most likely from recreating the optimizers.
*/
function resetCanvas() {
  const optimizers = {
    'adam': tf.train.adam(learningRate),
    'sgd': tf.train.sgd(learningRate)
  }

  tf.tidy(() => {
    m = tf.variable(tf.scalar(0));
    b = tf.variable(tf.scalar(0));

    optimizer = optimizers[optimizerSelect.value()];
  });

  X = [];
  Y = [];

  // Clear #data
  let data = document.getElementById('data');
  data.innerText = null;

  loop();
}
