new p5((sketch) => {
    const s = sketch

    const FRAME_RATE = 60

    s.preload = () => { }

    s.setup = () => {
        s.createCanvas(640, 480)
        s.frameRate(FRAME_RATE)
    }

    s.draw = async () => {
        // ---------------------------------------------------------------------
        s.background(0)

        s.strokeWeight(1)
        s.stroke(127)
        s.noFill()
        s.rect(0, 0, s.width, s.height)

        // ---------------------------------------------------------------------
        // TODO:
    }
}, 'p5-main')
