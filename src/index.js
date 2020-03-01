import * as tf from '@tensorflow/tfjs'

// import './p5/main'

(async () => {
    console.log('::: tfjs version:', tf.version)
    console.log('::: tensors:', tf.memory().numTensors)
    console.log('::: tfjs backend:', tf.getBackend())

    // Standard example ........................................................
    // Define a model for linear regression.
    // const model = tf.sequential()
    // model.add(tf.layers.dense({ units: 1, inputShape: [1] }))

    // model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' })

    // // Generate some synthetic data for training.
    // const xs = tf.tensor2d([1, 2, 3, 4], [4, 1])
    // const ys = tf.tensor2d([1, 3, 5, 7], [4, 1])

    // // Train the model using the data.
    // model.fit(xs, ys, { epochs: 10 }).then(() => {
    //     // Use the model to do inference on a data point the model hasn't seen before:
    //     model.predict(tf.tensor2d([5], [1, 1])).print()
    //     // Open the browser devtools to see the output
    // })

    // Some basic examples .....................................................
    // tf.tidy(() => {
    //     // scalar
    //     tf.tensor([87], []).print()
    //     tf.scalar(87).print()


    //     // 1D tensor
    //     tf.tensor([1, 2, 3], [3]).print()
    //     tf.tensor1d([1, 2, 3]).print()

    //     // 2D tensor
    //     tf.tensor([1, 2, 3, 4, 5, 6], [2, 3]).print()
    //     tf.tensor2d([[1, 2, 3], [4, 5, 6]]).print()

    //     // 3D tensor
    //     tf.tensor([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]).print()
    //     tf.tensor3d([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).print()

    //     const a = tf.tensor1d([1, 2, 3, 4])
    //     const b = tf.tensor1d([2, 2, 2, 2])
    //     a.add(b).print()
    //     a.print()
    // })

    // tf.tidy(() => {
    //     const xs = tf.tensor1d([1, 2, 3])
    //     const ys = xs.mul(tf.scalar(5))
    //     ys.print()
    // })

    // function getYs(xs, m, c) {
    //     const res = xs.mul(m).add(c)
    //     res.print()

    //     return res
    // }

    // tf.tidy(() => {
    //     getYs(tf.tensor1d([1, 5, 10]), 2, 1)
    // })

    // tf.tidy(() => {
    //     const t3 = tf.tensor1d([25, 76, 4, 23, -5, 22])
    //     const max = t3.max() // 76
    //     max.print()
    //     const min = t3.min() // -5
    //     min.print()

    //     t3.sub(min).div(max.sub(min)).print()
    // })

    // Linear regression example ...............................................
    const dataset = tf.data.csv('/data/kc_house_data.csv')
    const sampleDataset = dataset.take(10)
    const sampleArray = await sampleDataset.toArray()
    console.log('::: dataset:', sampleArray)


    console.log('::: tensors:', tf.memory().numTensors)
})()
