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

// function denormalizeMany(tensor, min, max) {
//     const dimensions = tensor.shape.length && tensor.shape[1]

//     if (!dimensions || dimensions === 1) {
//         return denormalizeOne(tensor, min, max)
//     }

//     const arrayOfTensors = tf.split(tensor, dimensions, 1)
//     const denormalized = arrayOfTensors.map((t, i) => denormalizeOne(t, min[i], max[i]))

//     return tf.concat(denormalized, 1)
// }

function getClassIndex(className) {
    if (className === '1') return 0
    if (className === '2') return 1
    return 2
}

function getClassName(classIndex) {
    switch (classIndex) {
        case 0: return '1'
        case 1: return '2'
    }

    return '3+'
}

async function plotPredictionHeatmap(
    model,
    normalizedFeatures,
    size = 400,
) {
    const valuesEtc = tf.tidy(() => {
        const gridSize = 50
        const predictionColumns = []
        for (let colIndex = 0; colIndex < gridSize; colIndex++) {
            const colInputs = []
            const x = colIndex / gridSize

            for (let rowIndex = 0; rowIndex < gridSize; rowIndex++) {
                const y = (gridSize - rowIndex) / gridSize
                colInputs.push([x, y])
            }

            const colPredictions = model.predict(tf.tensor2d(colInputs))
            predictionColumns.push(colPredictions)
        }

        const valuesTensor = tf.stack(predictionColumns)

        const normalizedTicksTensor = tf.linspace(0, 1, gridSize)
        const xTicksTensor = denormalizeOne(normalizedTicksTensor, normalizedFeatures.min[0], normalizedFeatures.max[0])
        const yTicksTensor = denormalizeOne(normalizedTicksTensor.reverse(), normalizedFeatures.min[1], normalizedFeatures.max[1])

        return [valuesTensor, xTicksTensor, yTicksTensor]
    })

    const values = await valuesEtc[0].array()

    const xTicks = await valuesEtc[1].array()
    const xTickLabels = xTicks.map((v) => (v / 1000).toFixed(1) + 'k sqft')

    const yTicks = await valuesEtc[2].array()
    const yTickLabels = yTicks.map((v) => '$' + (v / 1000).toFixed(0) + 'k')

    tf.unstack(values, 2).forEach((vals, i) => {
        const data = {
            values: vals,
            xTickLabels,
            yTickLabels,
        }

        tfvis.render.heatmap(
            {
                name: `Bedrooms: ${getClassName(i)} (local)`,
                tab: 'Predictions',
            },
            data,
            {
                colorMap: 'greyscale',
                height: size,
            },
        )

        tfvis.render.heatmap(
            {
                name: `Bedrooms: ${getClassName(i)} (full domain)`,
                tab: 'Predictions',
            },
            data,
            {
                colorMap: 'viridis',
                domain: [1, 0], // [0, 1]
                height: size,
            },
        )
    })
}

async function multiClassClassification() {
    tfvis.visor().toggleFullScreen()

    // Read data from CSV file
    const dataset = tf.data.csv('/data/kc_house_data.csv')

    // Extract data
    const pointsDataset = dataset
        .map((record) => ({
            x: record.sqft_living,
            y: record.price,
            class: record.bedrooms > 2 ? '3+' : '' + record.bedrooms,
        }))
        .filter(({ class: c }) => c !== '0')

    // Shuffle data
    const points = await pointsDataset.toArray()
    if (points.length % 2) {
        points.pop()
    }
    tf.util.shuffle(points)

    // Prepare features (inputs)
    const featureValues = points.map((point) => [point.x, point.y])
    const featureTensor = tf.tensor2d(featureValues)

    // Normalize features (min-max)
    const normalizedFeatures = normalizeMany(featureTensor)
    featureTensor.dispose()

    // Prepare labels (output)
    const labelValues = points.map((point) => getClassIndex(point.class))
    const labelTensor = tf.tidy(() => tf.oneHot(tf.tensor1d(labelValues, 'int32'), 3))

    // Slitting into training and testing features data
    const [trainingFeatureTensor, testingFeatureTensor] = tf.split(normalizedFeatures.tensor, 2)

    // Slitting into training and testing label data
    const [trainingLabelTensor, testingLabelTensor] = tf.split(labelTensor, 2)

    // Check if the model exists
    const models = await tf.io.listModels()
    const modelInfo = models['localstorage://mclass']
    console.log('::: Model info:', modelInfo)
    let model
    if (!modelInfo) {
        // Create model
        model = tf.sequential()

        model.add(tf.layers.dense({
            units: 10,
            useBias: true,
            activation: 'sigmoid',
            inputDim: 2,
        }))
        model.add(tf.layers.dense({
            units: 10,
            useBias: true,
            activation: 'sigmoid',
        }))
        model.add(tf.layers.dense({
            units: 3,
            useBias: true,
            activation: 'softmax',
        }))

        const optimizer = tf.train.adam()
        model.compile({
            loss: 'binaryCrossentropy',
            optimizer,
            metrics: ['accuracy'],
        })

        model.compile({
            loss: 'categoricalCrossentropy',
            optimizer: 'adam',
        })

        // Train model
        const {
            // onBatchEnd,
            onEpochEnd,
        } = tfvis.show.fitCallbacks(
            { name: 'Training Performance' },
            ['loss', 'val_loss'],
        )

        const trainingResult = await model.fit(trainingFeatureTensor, trainingLabelTensor, {
            batchSize: 32, // 1024, default: 32
            epochs: 150, // Overfitting?
            validationSplit: 0.2,
            callbacks: {
                // onBatchEnd,
                onEpochBegin: () => {
                    plotPredictionHeatmap(model, normalizedFeatures)
                },
                onEpochEnd: (epoch, log) => {
                    console.log(`::: Epoch ${epoch}: loss = ${log.loss.toFixed(5)} (${log.val_loss.toFixed(5)})`)
                    onEpochEnd(epoch, log)
                },
            },
        })
        const trainingLoss = [...trainingResult.history.loss].pop()
        const validationLoss = [...trainingResult.history.val_loss].pop()
        console.log(`::: Training (Validation) loss: ${trainingLoss.toFixed(5)} (${validationLoss.toFixed(5)})`)

        // Testing model
        const testingResult = await model.evaluate(testingFeatureTensor, testingLabelTensor).dataSync()
        // console.log('::: Testing result:', testingResult)
        const testingLoss = testingResult[0]
        console.log('::: Testing loss:', testingLoss.toFixed(5))

        // Save model
        const saveResults = await model.save('localstorage://mclass')
        console.log('::: Save results:', saveResults)
    }
    else {
        // Load model
        model = await tf.loadLayersModel('localstorage://mclass')
    }

    // Inspect model
    model.summary()
    tfvis.show.modelSummary({ name: 'Model Summary' }, model)

    //     // Make prediction
    //     const inputLivingSpaceValue = 2000
    //     const inputHousePriceValue = 3000000
    //     const inputTensor = tf.tensor2d([[inputLivingSpaceValue, inputHousePriceValue]])
    //     const normalizedInput = normalizeMany(inputTensor, normalizedFeatures.min, normalizedFeatures.max, ':NOW:')
    //     const outputTensor = model.predict(normalizedInput.tensor)
    //     const outputValue = outputTensor.dataSync()
    //     console.log('::: Predicted output value:', (outputValue * 100).toFixed(1), '%')

    // Visualize data
    await plotPredictionHeatmap(model, normalizedFeatures)

    const allSeries = points.reduce((acc, point) => {
        const name = `Bedrooms: ${point.class}`

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

export default multiClassClassification
