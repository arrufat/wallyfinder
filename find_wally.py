#!/usr/bin/env python

# from build.Release.wallyfinder import WallyFinder
from wallyfinder import WallyFinder
from PIL import Image, ImageDraw
from sys import argv

colors = {"wally": "cyan"}
wf = WallyFinder()

image = Image.open(argv[1]).convert("RGB")

dets = wf(image)

draw = ImageDraw.Draw(image)
for i, det in enumerate(dets):
    draw.rectangle(
        ((det["xmin"], det["ymin"]), (det["xmax"], det["ymax"])),
        outline=colors[det["label"]],
        width=5,
    )
    # draw.text(
    #     (det["xmin"] + 2, det["ymin"] + 1),
    #     "{:02d}: {} ({:.2})".format(i, det["label"], det["confidence"]),
    #     fill="black",
    # )

image.show()
image.save("solution.jpg")
