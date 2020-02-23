async function request(resource = '', init, custom) {
    const {
        method = 'GET',
        headers = {},
        data,
        body,
        mode = 'cors', // no-cors, cors, *same-origin
        // credentials: 'same-origin', // include, *same-origin, omit
        // cache: 'no-cache', // *default, no-cache, reload, force-cache, only-if-cached
        // redirect: 'follow', // manual, *follow, error
        // referrer: 'no-referrer', // no-referrer, *client
    } = init || {}

    const {
        cb = () => {},
    } = custom || {}

    let payload
    if (!['OPTIONS', 'GET', 'HEAD'].includes(method.toUpperCase())) {
        if (body) payload = body
        else if (data) payload = JSON.stringify(data)
    }

    console.log('TODO::: >>> request:', { resource, method, payload: body || data })

    const combinedHeaders = {
        'Content-Type': headers['Content-Type'] || 'application/json',
        ...headers,
    }

    let response
    try {
        response = await fetch(resource, {
            method,
            headers: combinedHeaders,
            body: payload,
            mode,
        })
    }
    catch (reason) {
        console.error('::: [request/fetch] reason:', reason)
        throw reason
    }

    // if (response.ok ... response.status < 200 || response.status >= 300) ...

    let responseData
    try {
        if (combinedHeaders['Content-Type'] === 'application/json') {
            responseData = await response.json()
        }
        // else if ...
    }
    catch (reason) {
        console.error('::: [request/parse] reason:', reason)
        throw reason
    }

    cb(response, responseData)

    return {
        response,
        data: responseData,
    }
}

export default request
