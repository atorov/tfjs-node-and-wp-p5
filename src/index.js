import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'

// import './p5/main'

console.log('::: tfjs version:', tf.version)
console.log('::: tensors:', tf.memory().numTensors)
console.log('::: tfjs backend:', tf.getBackend());

(async () => {
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
    // function normalize(tensor, min = tensor.min(), max = tensor.max()) {
    //     return tf.tidy(() => {
    //         const normalizedTensor = tensor.sub(min).div(max.sub(min))

    //         return {
    //             tensor: normalizedTensor,
    //             min,
    //             max,
    //         }
    //     })
    // }

    // function denormalize(tensor, min, max) {
    //     return tensor.mul(max.sub(min)).add(min)
    // }

    // // Read data from CSV file
    // const dataset = tf.data.csv('/data/kc_house_data.csv')
    // // const sampleDataset = dataset.take(10)
    // // const sampleArray = await sampleDataset.toArray()
    // // console.log('::: dataset:', sampleArray)

    // // Extract data
    // const pointsDataset = dataset.map((record) => ({
    //     x: record.sqft_living,
    //     y: record.price,
    // }))

    // // Shuffle data
    // const points = await pointsDataset.toArray()
    // if (points.length % 2) {
    //     points.pop()
    // }
    // tf.util.shuffle(points)

    // // Prepare features (inputs)
    // const featureValues = points.map((point) => point.x)
    // const featureTensor = tf.tensor2d(featureValues, [featureValues.length, 1])

    // // Prepare labels (output)
    // const labelValues = points.map((point) => point.y)
    // const labelTensor = tf.tensor2d(labelValues, [labelValues.length, 1])

    // // Normalize features (min-max)
    // const normalizedFeatures = normalize(featureTensor)
    // featureTensor.dispose()

    // // Normalize labels (min-max)
    // const normalizedLabels = normalize(labelTensor)
    // labelTensor.dispose()

    // // Slitting into training and testing features data
    // const [trainingFeatureTensor, testingFeatureTensor] = tf.split(normalizedFeatures.tensor, 2)

    // // Slitting into training and testing label data
    // const [trainingLabelTensor, testingLabelTensor] = tf.split(normalizedLabels.tensor, 2)

    // // Check if the model exists
    // const models = await tf.io.listModels()
    // const modelInfo = models['localstorage://linreg']
    // console.log('::: Model info:', modelInfo)
    // let model
    // if (!modelInfo) {
    //     // Create model
    //     model = tf.sequential()
    //     model.add(tf.layers.dense({
    //         units: 1,
    //         useBias: true, // default: true
    //         activation: 'linear', // default: 'linear'
    //         inputDim: 1,
    //     }))

    //     model.compile({
    //         loss: 'meanSquaredError',
    //         // optimizer: tf.train.sgd(0.1),
    //         optimizer: 'adam',
    //     })

    //     // Train model
    //     const {
    //         // onBatchEnd,
    //         onEpochEnd,
    //     } = tfvis.show.fitCallbacks(
    //         { name: 'Training Performance' },
    //         ['loss', 'val_loss'],
    //     )

    //     const trainingResult = await model.fit(trainingFeatureTensor, trainingLabelTensor, {
    //         batchSize: 32, // 1024, default: 32
    //         epochs: 20,
    //         validationSplit: 0.2,
    //         callbacks: {
    //             // onBatchEnd,
    //             onEpochEnd: (epoch, log) => {
    //                 console.log(`::: Epoch ${epoch}: loss = ${log.loss.toFixed(5)} (${log.val_loss.toFixed(5)})`)
    //                 onEpochEnd(epoch, log)
    //             },
    //         },
    //     })

    //     // console.log('::: Training result:', result)
    //     const trainingLoss = [...trainingResult.history.loss].pop()
    //     const validationLoss = [...trainingResult.history.val_loss].pop()
    //     console.log(`::: Training (Validation) loss: ${trainingLoss.toFixed(5)} (${validationLoss.toFixed(5)})`)

    //     // Testing model
    //     const testingResult = await model.evaluate(testingFeatureTensor, testingLabelTensor).dataSync()
    //     // console.log('::: Testing result:', testingResult)
    //     const testingLoss = testingResult[0]
    //     console.log('::: Testing loss:', testingLoss.toFixed(5))

    //     // Save model
    //     const saveResults = await model.save('localstorage://linreg')
    //     console.log('::: Save results:', saveResults)
    // }
    // else {
    //     // Load model
    //     model = await tf.loadLayersModel('localstorage://linreg')
    // }

    // // Inspect model
    // model.summary()
    // tfvis.show.modelSummary({ name: 'Model Summary' }, model)
    // const layer = model.getLayer(null, 0)
    // tfvis.show.layer({ name: 'Layer 1' }, layer)

    // // Make prediction
    // tf.tidy(() => {
    //     const inputValue = 4000
    //     const inputTensor = tf.tensor1d([inputValue])
    //     const normalizedInput = normalize(inputTensor, normalizedFeatures.min, normalizedFeatures.max)
    //     const normalizedOutput = model.predict(normalizedInput.tensor)
    //     const outputTensor = denormalize(normalizedOutput, normalizedLabels.min, normalizedLabels.max)
    //     const outputValue = outputTensor.dataSync()[0]
    //     console.log('::: Predicted output value: $', (outputValue / 1e6).toFixed(3), 'M')
    // })

    // // Visualize data
    // tf.tidy(() => {
    //     const normalizedXs = tf.linspace(0, 1, 100)
    //     const xs = denormalize(normalizedXs, normalizedFeatures.min, normalizedFeatures.max).dataSync()

    //     const normalizedYs = model.predict(normalizedXs.reshape([100, 1]))
    //     const ys = denormalize(normalizedYs, normalizedLabels.min, normalizedLabels.max).dataSync()

    //     const predictions = Array.from(xs).map((x, index) => ({
    //         x,
    //         y: ys[index],
    //     }))

    //     tfvis.render.scatterplot(
    //         {
    //             name: 'Square feet vs House Price',
    //         },
    //         {
    //             values: [points, predictions],
    //             series: ['original', 'predicted'],
    //         },
    //         {
    //             xLabel: 'Square feet',
    //             yLabel: 'Price',
    //         },
    //     )
    // })

    // Non-linear regression example ...............................................
    // function normalize(tensor, min = tensor.min(), max = tensor.max()) {
    //     return tf.tidy(() => {
    //         const normalizedTensor = tensor.sub(min).div(max.sub(min))

    //         return {
    //             tensor: normalizedTensor,
    //             min,
    //             max,
    //         }
    //     })
    // }

    // function denormalize(tensor, min, max) {
    //     return tensor.mul(max.sub(min)).add(min)
    // }

    // // Read data from CSV file
    // const dataset = tf.data.csv('/data/kc_house_data.csv')

    // // Extract data
    // const pointsDataset = dataset.map((record) => ({
    //     x: record.sqft_living,
    //     y: record.price,
    // }))

    // // Shuffle data
    // const points = await pointsDataset.toArray()
    // if (points.length % 2) {
    //     points.pop()
    // }
    // tf.util.shuffle(points)

    // // Prepare features (inputs)
    // const featureValues = points.map((point) => point.x)
    // const featureTensor = tf.tensor2d(featureValues, [featureValues.length, 1])

    // // Prepare labels (output)
    // const labelValues = points.map((point) => point.y)
    // const labelTensor = tf.tensor2d(labelValues, [labelValues.length, 1])

    // // Normalize features (min-max)
    // const normalizedFeatures = normalize(featureTensor)
    // featureTensor.dispose()

    // // Normalize labels (min-max)
    // const normalizedLabels = normalize(labelTensor)
    // labelTensor.dispose()

    // // Slitting into training and testing features data
    // const [trainingFeatureTensor, testingFeatureTensor] = tf.split(normalizedFeatures.tensor, 2)

    // // Slitting into training and testing label data
    // const [trainingLabelTensor, testingLabelTensor] = tf.split(normalizedLabels.tensor, 2)

    // // Check if the model exists
    // const models = await tf.io.listModels()
    // const modelInfo = models['localstorage://nlinreg']
    // console.log('::: Model info:', modelInfo)
    // let model
    // if (!modelInfo) {
    //     // Create model
    //     model = tf.sequential()
    //     model.add(tf.layers.dense({
    //         units: 3,
    //         activation: 'sigmoid',
    //         inputDim: 1,
    //     }))
    //     model.add(tf.layers.dense({
    //         units: 1,
    //         activation: 'sigmoid',
    //     }))

    //     model.compile({
    //         loss: 'meanSquaredError',
    //         optimizer: 'adam',
    //     })

    //     // Train model
    //     const {
    //         // onBatchEnd,
    //         onEpochEnd,
    //     } = tfvis.show.fitCallbacks(
    //         { name: 'Training Performance' },
    //         ['loss', 'val_loss'],
    //     )

    //     const trainingResult = await model.fit(trainingFeatureTensor, trainingLabelTensor, {
    //         batchSize: 32, // 1024, default: 32
    //         epochs: 100,
    //         validationSplit: 0.2,
    //         callbacks: {
    //             // onBatchEnd,
    //             onEpochEnd: (epoch, log) => {
    //                 console.log(`::: Epoch ${epoch}: loss = ${log.loss.toFixed(5)} (${log.val_loss.toFixed(5)})`)
    //                 onEpochEnd(epoch, log)
    //             },
    //         },
    //     })
    //     const trainingLoss = [...trainingResult.history.loss].pop()
    //     const validationLoss = [...trainingResult.history.val_loss].pop()
    //     console.log(`::: Training (Validation) loss: ${trainingLoss.toFixed(5)} (${validationLoss.toFixed(5)})`)

    //     // Testing model
    //     const testingResult = await model.evaluate(testingFeatureTensor, testingLabelTensor).dataSync()
    //     // console.log('::: Testing result:', testingResult)
    //     const testingLoss = testingResult[0]
    //     console.log('::: Testing loss:', testingLoss.toFixed(5))

    //     // Save model
    //     const saveResults = await model.save('localstorage://nlinreg')
    //     console.log('::: Save results:', saveResults)
    // }
    // else {
    //     // Load model
    //     model = await tf.loadLayersModel('localstorage://nlinreg')
    // }

    // // Inspect model
    // model.summary()
    // tfvis.show.modelSummary({ name: 'Model Summary' }, model)
    // const layer = model.getLayer(null, 0)
    // tfvis.show.layer({ name: 'Layer 1' }, layer)

    // // Make prediction
    // tf.tidy(() => {
    //     const inputValue = 4000
    //     const inputTensor = tf.tensor1d([inputValue])
    //     const normalizedInput = normalize(inputTensor, normalizedFeatures.min, normalizedFeatures.max)
    //     const normalizedOutput = model.predict(normalizedInput.tensor)
    //     const outputTensor = denormalize(normalizedOutput, normalizedLabels.min, normalizedLabels.max)
    //     const outputValue = outputTensor.dataSync()[0]
    //     console.log('::: Predicted output value: $', (outputValue / 1e6).toFixed(3), 'M')
    // })

    // // Visualize data
    // tf.tidy(() => {
    //     const normalizedXs = tf.linspace(0, 1, 100)
    //     const xs = denormalize(normalizedXs, normalizedFeatures.min, normalizedFeatures.max).dataSync()

    //     const normalizedYs = model.predict(normalizedXs.reshape([100, 1]))
    //     const ys = denormalize(normalizedYs, normalizedLabels.min, normalizedLabels.max).dataSync()

    //     const predictions = Array.from(xs).map((x, index) => ({
    //         x,
    //         y: ys[index],
    //     }))

    //     tfvis.render.scatterplot(
    //         {
    //             name: 'Square feet vs House Price',
    //         },
    //         {
    //             values: [points, predictions],
    //             series: ['original', 'predicted'],
    //         },
    //         {
    //             xLabel: 'Square feet',
    //             yLabel: 'Price',
    //         },
    //     )
    // })

    // Binary classification ...................................................
    // function normalize(tensor, min = tensor.min(), max = tensor.max()) {
    //     return tf.tidy(() => {
    //         const normalizedTensor = tensor.sub(min).div(max.sub(min))

    //         return {
    //             tensor: normalizedTensor,
    //             min,
    //             max,
    //         }
    //     })
    // }

    // function denormalize(tensor, min, max) {
    //     return tensor.mul(max.sub(min)).add(min)
    // }

    // Read data from CSV file
    const dataset = tf.data.csv('/data/kc_house_data.csv')

    // Extract data
    const pointsDataset = dataset.map((record) => ({
        x: record.sqft_living,
        y: record.price,
        class: record.waterfront,
    }))

    // Shuffle data
    const points = await pointsDataset.toArray()
    if (points.length % 2) {
        points.pop()
    }
    tf.util.shuffle(points)

    // // Prepare features (inputs)
    // const featureValues = points.map((point) => point.x)
    // const featureTensor = tf.tensor2d(featureValues, [featureValues.length, 1])

    // // Prepare labels (output)
    // const labelValues = points.map((point) => point.y)
    // const labelTensor = tf.tensor2d(labelValues, [labelValues.length, 1])

    // // Normalize features (min-max)
    // const normalizedFeatures = normalize(featureTensor)
    // featureTensor.dispose()

    // // Normalize labels (min-max)
    // const normalizedLabels = normalize(labelTensor)
    // labelTensor.dispose()

    // // Slitting into training and testing features data
    // const [trainingFeatureTensor, testingFeatureTensor] = tf.split(normalizedFeatures.tensor, 2)

    // // Slitting into training and testing label data
    // const [trainingLabelTensor, testingLabelTensor] = tf.split(normalizedLabels.tensor, 2)

    // // Check if the model exists
    // const models = await tf.io.listModels()
    // const modelInfo = models['localstorage://nlinreg']
    // console.log('::: Model info:', modelInfo)
    // let model
    // if (!modelInfo) {
    //     // Create model
    //     model = tf.sequential()
    //     model.add(tf.layers.dense({
    //         units: 3,
    //         activation: 'sigmoid',
    //         inputDim: 1,
    //     }))
    //     model.add(tf.layers.dense({
    //         units: 1,
    //         activation: 'sigmoid',
    //     }))

    //     model.compile({
    //         loss: 'meanSquaredError',
    //         optimizer: 'adam',
    //     })

    //     // Train model
    //     const {
    //         // onBatchEnd,
    //         onEpochEnd,
    //     } = tfvis.show.fitCallbacks(
    //         { name: 'Training Performance' },
    //         ['loss', 'val_loss'],
    //     )

    //     const trainingResult = await model.fit(trainingFeatureTensor, trainingLabelTensor, {
    //         batchSize: 32, // 1024, default: 32
    //         epochs: 100,
    //         validationSplit: 0.2,
    //         callbacks: {
    //             // onBatchEnd,
    //             onEpochEnd: (epoch, log) => {
    //                 console.log(`::: Epoch ${epoch}: loss = ${log.loss.toFixed(5)} (${log.val_loss.toFixed(5)})`)
    //                 onEpochEnd(epoch, log)
    //             },
    //         },
    //     })
    //     const trainingLoss = [...trainingResult.history.loss].pop()
    //     const validationLoss = [...trainingResult.history.val_loss].pop()
    //     console.log(`::: Training (Validation) loss: ${trainingLoss.toFixed(5)} (${validationLoss.toFixed(5)})`)

    //     // Testing model
    //     const testingResult = await model.evaluate(testingFeatureTensor, testingLabelTensor).dataSync()
    //     // console.log('::: Testing result:', testingResult)
    //     const testingLoss = testingResult[0]
    //     console.log('::: Testing loss:', testingLoss.toFixed(5))

    //     // Save model
    //     const saveResults = await model.save('localstorage://nlinreg')
    //     console.log('::: Save results:', saveResults)
    // }
    // else {
    //     // Load model
    //     model = await tf.loadLayersModel('localstorage://nlinreg')
    // }

    // // Inspect model
    // model.summary()
    // tfvis.show.modelSummary({ name: 'Model Summary' }, model)
    // const layer = model.getLayer(null, 0)
    // tfvis.show.layer({ name: 'Layer 1' }, layer)

    // // Make prediction
    // tf.tidy(() => {
    //     const inputValue = 4000
    //     const inputTensor = tf.tensor1d([inputValue])
    //     const normalizedInput = normalize(inputTensor, normalizedFeatures.min, normalizedFeatures.max)
    //     const normalizedOutput = model.predict(normalizedInput.tensor)
    //     const outputTensor = denormalize(normalizedOutput, normalizedLabels.min, normalizedLabels.max)
    //     const outputValue = outputTensor.dataSync()[0]
    //     console.log('::: Predicted output value: $', (outputValue / 1e6).toFixed(3), 'M')
    // })

    // Visualize data
    tf.tidy(() => {
        tfvis.visor().toggleFullScreen()

        //     const normalizedXs = tf.linspace(0, 1, 100)
        //     const xs = denormalize(normalizedXs, normalizedFeatures.min, normalizedFeatures.max).dataSync()

        //     const normalizedYs = model.predict(normalizedXs.reshape([100, 1]))
        //     const ys = denormalize(normalizedYs, normalizedLabels.min, normalizedLabels.max).dataSync()

        //     const predictions = Array.from(xs).map((x, index) => ({
        //         x,
        //         y: ys[index],
        //     }))

        const allSeries = points.reduce((acc, point) => {
            const name = `Waterfront: ${point.class}`

            return {
                ...acc,
                [name]: [...(acc[name] || []), point],
            }
        }, {})

        tfvis.render.scatterplot(
            {
                name: 'Square feet vs House Price',
                styles: {
                    width: '100%',
                    height: '100%',
                },
            },
            {
                values: Object.values(allSeries),
                series: Object.keys(allSeries),
            },
            {
                xLabel: 'Square feet',
                yLabel: 'Price',
                width: 800,
                height: 600,
            },
        )
    })

    // .........................................................................
    console.log('::: tensors:', tf.memory().numTensors)
})()
