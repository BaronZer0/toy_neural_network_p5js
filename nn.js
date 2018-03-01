function sigmoid(z) {
  return 1 / (1 + Math.exp(-z));
}

function dsigmoid(z) {
  return sigmoid(z) * (1 - sigmoid(z));
}

class NeuralNetwork {
  constructor(layers) {
    this.l = layers;
    this.w = [];
    this.b = [];
    //create zero matrices for wheights and biases
    for (let i = 0; i < this.l.length - 1; i++)
      this.w[i] = math.zeros([this.l[i+1],this.l[i]]);
    for (let i = 0; i < this.l.length - 1; i++)
      this.b[i] = math.zeros([this.l[i+1]]);
    //fill matrices with random values
    this.w = this.w.map(w => {w = w.map(x => x = x.map(r => random(-1,1))); return w});
    this.b = this.w.map(b => {b = b.map(r => random(-1,1)); return b});
  }

  feedforward(input) {
    let a = [];
    let z = [];
    a[0] = input;
    for (let i = 0; i < this.l.length-1; i++) {
      z[i] = math.multiply(this.w[i], math.transpose(a[i]));
      z[i] = math.add(z[i], this.b[i]);
      a[i+1] = z[i].map(sigmoid);
    }
    return [a, z];
  }

  predict(input) {
    return this.feedforward(input)[0][this.l.length-1];
  }

  evaluate(test_data) {
    return math.subtract(this.predict(test_data.x), test_data.y);
  }

  //stochastic gradient descent applying backpropagation in mini-batches
  //currently unused
  SGD(training_data, epochs, batch_size, lr, test_data = null) {
    for (let i = 0; i < epochs; i++) {
      shuffle(training_data);
      let mini_batches = new Array();
      for(let k = 0; k < training_data.length / batch_size; k++) {
        mini_batches.push(subset(training_data, k, k+batch_size));
      }
      for(mini_batch in mini_batches) {
        this.update_mini_batch(mini_batch, lr);
        if(test_data) {
          console.log("Epoch" + i + " ,scored: " + this.evaluate(test_data) + "%")
        } else {
          console.log("Epoch" + i + " complete");
        }
      }
    }
  }

  update_mini_batch(mini_batch, lr) {
    let self = this;
    let nabla_w = new Array(), nabla_b = new Array();
    this.w.forEach(function(w) {nabla_w.push(math.zeros(math.size(w)))});
    this.b.forEach(function(b) {nabla_b.push(math.zeros(math.size(b)))});
    mini_batch.forEach(function(data) {
      let delta_nabla_w, delta_nabla_b;
      [delta_nabla_w, delta_nabla_b] = self.backpropagate(data.x, data.y);
      nabla_w = nabla_w.map(function(nw, i) {
        return math.add(nabla_w[i], delta_nabla_w[i])
      });
      nabla_b = nabla_b.map(function(nb, i) {
        return math.add(nabla_b[i], delta_nabla_b[i])
      });
      self.w = self.w.map(function(w, i) {
        return math.subtract(w, math.multiply((lr / mini_batch.length), nabla_w[i]));
      });
      self.b = self.b.map(function(b, i) {
        return math.subtract(b, math.multiply((lr / mini_batch.length), nabla_b[i]));
      });
    });
    this.w = self.w;
    this.b = self.b;
  }


  train(data, lr) {
    let nabla_w, nabla_b;
    [nabla_w, nabla_b] = this.backpropagate(data.x, data.y);
    //console.table(nabla_w);
    this.w = this.w.map(function(w, i) {
      return math.subtract(w, math.multiply(lr, nabla_w[i]));
    });
  }

  backpropagate(x, y) {
    let nabla_w = new Array(), nabla_b = new Array();
    this.w.forEach(function(w) {nabla_w.push(math.zeros(math.size(w)))});
    this.b.forEach(function(b) {nabla_b.push(math.zeros(math.size(b)))});
    let a = [], z = [];
    [a, z] = this.feedforward(x);
    let l = this.l.length-1;
    let dC_da = math.subtract(a[l], y);
    let da_dz = z[l-1].map(dsigmoid);
    let delta = math.dotMultiply(dC_da, da_dz);
    nabla_w[l-1] = math.multiply(math.transpose([delta]), [a[l-1]]);
    nabla_b[l-1] = delta;
    for(let i = l-1; i > 0; i--) {
      da_dz = z[i].map(dsigmoid);
      delta = math.multiply(math.transpose(this.w[i]), delta);
      nabla_w[i-1] = math.multiply(math.transpose([delta]), [a[i-1]]);
      nabla_b[i-1] = delta;
    }
    return [nabla_w, nabla_b];
  }
}

class NNVisualizer {
  constructor(layers) {
    this.grid = [80,40];
    this.orig = [20,20];
    this.nodes = new Array();
  	for(let i = 0; i < layers.length; i++) {
  		this.nodes.push([]);
  		for(let j = 0; j < layers[i]; j++) {
        let x = this.orig[0] + i * this.grid[0];
        let y = this.orig[1] + (j + (max(layers)-layers[i])/2) * this.grid[1];
  			this.nodes[i][j] = new Node(x, y);
  		}
  	}
  }

  draw(nn, input) {
    let nodes = this.nodes;
    //draw wheights
    for(let i = 0; i < nn.w.length; i++) {
  		math.forEach(nn.w[i], function (value, index) {
  			let n1 = nodes[i][index[1]];
  			let n2 = nodes[i+1][index[0]];
  			strokeWeight(value);
        if(value > 0) {
          stroke("green");
        } else {
          stroke("red");
        }
  			line(n1.x, n1.y, n2.x, n2.y);
  		});
  	}
    //draw nodes
  	strokeWeight(0.5);
    stroke(0);
    let a, z;
    [a, z] = nn.feedforward(input);
  	for(let i = 0; i < a.length; i++) {
  		math.forEach(a[i], function (value, j) {
				nodes[i][j].activate(value);
  		});
  	}
  }
}

class Node {
  constructor(x, y) {
    this.x = x;
    this.y = y;
  }

  activate(a) {
    fill(color(a * 255));
    ellipse(this.x, this.y, 25, 25);
    fill(color(round(map(a,0,1,1,0)) * 255));
    text(math.round(a,1), this.x-8, this.y+5);
  }
}
