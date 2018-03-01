function setup() {
	createCanvas(600, 600);
	var layers = [2,4,1];
	this.nn = new NeuralNetwork(layers);
	this.nnv = new NNVisualizer(layers);
	this.input = new Array(layers[0]);
	for(let i = 0; i < this.input.length; i++) {
		this.input[i] = round(random(0,1));
	}
	var f = nn.feedforward([0,1]);
	//console.table(f[0]);
	//nn.backpropagate([0,1], [1,0]);
}

function windowResized() {
  //resizeCanvas(windowWidth, windowHeight);
}

function draw() {
	background(255);

	//create data batch
	var batch = new Array(100);
	batch = batch.fill(0).map(function(b) {
		let input = [random(), random()];
		let xor = abs(input[0] - input[1]);
		let output = [xor];
		return {x: input, y: output};
	});
	var data = batch[round(random(batch.size))];

	let lr = 0.1;
	//use the data batch to train at once, averaging the changes
	this.nn.update_mini_batch(batch, lr);

	//use one data point to train and apply a change
	//this.nn.train(data, lr);

	//visualization
	io_map2d(0, 0, 1, 1);
	this.nnv.draw(this.nn, data.x);
	var e = this.nn.evaluate(data)[0];
	text(math.round(abs(e),4), 0, 10);
}

function io_map2d(x1, y1, x2, y2, res=10) {
	let cols = width / res;
	let rows = width / res;
	for(let i = 0; i < cols; i++) {
		for(let j = 0; j < rows; j++) {
			let x = x1 + i/cols * (x2 - x1);
			let y = y1 + j/rows * (y2 - x1);
			let p = nn.predict([x, y])[0];
			strokeWeight(0);
			fill(p*255);
			rect(i*res, j*res, res, res);
		}
	}
}
