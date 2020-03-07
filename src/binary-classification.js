import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'

function normalizeOne(tensor, min = tensor.min(), max = tensor.max()) {
    return {
        tensor: tensor.sub(min).div(max.sub(min)),
        min,
        max,
    }
}

function normalizeMany(tensor, min, max) {
    const dimensions = tensor.shape.length && tensor.shape[1]

    if (!dimensions || dimensions === 1) {
        return normalizeOne(tensor, min, max)
    }

    const arrayOfTensors = tf.split(tensor, dimensions, 1)

    const normalized = arrayOfTensors.map((t, i) => normalizeOne(
        t,
        min ? min[i] : undefined,
        max ? max[i] : undefined,
    ))

    const normalizedTensors = normalized.map(({ tensor: t }) => t)
    const returnTensor = tf.concat(normalizedTensors, 1)

    return {
        tensor: returnTensor,
        min: normalized.map(({ min: m }) => m),
        max: normalized.map(({ max: m }) => m),
    }
}

function denormalizeOne(tensor, min, max) {
    return tensor.mul(max.sub(min)).add(min)
}

function denormalizeMany(tensor, min, max) {
    const dimensions = tensor.shape.length && tensor.shape[1]

    if (!dimensions || dimensions === 1) {
        return denormalizeOne(tensor, min, max)
    }

    const arrayOfTensors = tf.split(tensor, dimensions, 1)
    const denormalized = arrayOfTensors.map((t, i) => denormalizeOne(t, min[i], max[i]))

    return tf.concat(denormalized, 1)
}

async function binaryClassification() {
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

    // Prepare features (inputs)
    const featureValues = points.map((point) => [point.x, point.y])
    const featureTensor = tf.tensor2d(featureValues)

    // Prepare labels (output)
    const labelValues = points.map((point) => point.class)
    const labelTensor = tf.tensor2d(labelValues, [labelValues.length, 1])

    // Normalize features (min-max)
    const normalizedFeatures = normalizeMany(featureTensor)
    featureTensor.dispose()

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
    //     const inputValue = 4000
    //     const inputTensor = tf.tensor1d([inputValue])
    //     const normalizedInput = normalize(inputTensor, normalizedFeatures.min, normalizedFeatures.max)
    //     const normalizedOutput = model.predict(normalizedInput.tensor)
    //     const outputTensor = denormalize(normalizedOutput, normalizedLabels.min, normalizedLabels.max)
    //     const outputValue = outputTensor.dataSync()[0]
    //     console.log('::: Predicted output value: $', (outputValue / 1e6).toFixed(3), 'M')

    // Visualize data
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
}

export default binaryClassification
