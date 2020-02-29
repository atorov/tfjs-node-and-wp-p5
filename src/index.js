import * as tf from '@tensorflow/tfjs'

// import './p5/main'

console.log('::: tfjs version:', tf.version)

// Define a model for linear regression.
const model = tf.sequential()
model.add(tf.layers.dense({ units: 1, inputShape: [1] }))

model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' })

// Generate some synthetic data for training.
const xs = tf.tensor2d([1, 2, 3, 4], [4, 1])
const ys = tf.tensor2d([1, 3, 5, 7], [4, 1])

// Train the model using the data.
model.fit(xs, ys, { epochs: 10 }).then(() => {
    // Use the model to do inference on a data point the model hasn't seen before:
    model.predict(tf.tensor2d([5], [1, 1])).print()
    // Open the browser devtools to see the output
})

console.log('::: tfjs backend:', tf.getBackend())

tf.tidy(() => {
    // scalar
    tf.tensor([87], []).print()
    tf.scalar(87).print()


    // 1D tensor
    tf.tensor([1, 2, 3], [3]).print()
    tf.tensor1d([1, 2, 3]).print()

    // 2D tensor
    tf.tensor([1, 2, 3, 4, 5, 6], [2, 3]).print()
    tf.tensor2d([[1, 2, 3], [4, 5, 6]]).print()

    // 3D tensor
    tf.tensor([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]).print()
    tf.tensor3d([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).print()

    const a = tf.tensor1d([1, 2, 3, 4])
    const b = tf.tensor1d([2, 2, 2, 2])
    a.add(b).print()
    a.print()
})

setTimeout(() => {
    console.log('::: tensors:', tf.memory().numTensors)

    tf.tidy(() => {
        for (let i = 0; i < 1000; i += 3) {
            const ts = tf.scalar(2)
            const t1 = tf.tensor1d([i, i + 1, i + 2])
            t1.mul(ts)
        }
    })

    console.log('::: tensors:', tf.memory().numTensors)
}, 2550)
