extensions [ py matrix profiler vid bitmap ]

globals [
 leftbox
 rightbox
 preconv ;; matrix of greyscale values
 postconv ;; matrix of post-conv values
 postprocessed ;; matrix of processed post-conv --> greyscale values
]

breed [tracers tracer]
breed [kerns kern]
breed [zoomies zoomie]

tracers-own [
  resides
]

patches-own [
  key?
  points-at
]

to startup
  py:setup "C:/Users/Matt/anaconda3/python"
end

;; DISPLAY RELATED ;;

to setup
  clear-all
  set leftbox false ; empty
  set rightbox false ; empty
  resize-world 0 599 0 249 ;; 600 W x 250 H
  reset-tracer
  ask patches [ set pcolor 69 ]
  ask patches with [ ((pxcor >= 25 and pxcor <= 224) or (pxcor >= 375 and pxcor <= 574)) and (pycor >= 25 and pycor <= 224) ] [
    set pcolor 49 ; pre-conv image and post-conv image display box (200 W x 200 H each)
  ]
  ;ask patches with [ pxcor >= 245 and pxcor <= 354 and pycor >= 45 and pycor <= 154 ] [
  ;  set pcolor 69 ; kernel display box (100 W x 100 H)
  ;]
end

to import-pic
  ;; Get image, grayscale it, resize it to fit in 200 x 200 square, get its matrix
  clear-globals
  py:set "f" user-file ;; path to the image
  (py:run
   "from PIL import Image, ImageOps"
   "import numpy as np"
   "import math"
   "image = Image.open(f)"
   "image = ImageOps.grayscale(image)"
   "width, height = image.size"
   "if max(width, height) >= 200: image.thumbnail((200, 200))" ;; resize and preserve aspect ratio
   "elif width > height: image = image.resize((200, math.ceil(200 / width * height)))"
   "else: image = image.resize((math.ceil(200 / height * width), 200))" ;; height > width
   "arr = np.array(image)"
  )
  set preconv matrix:from-row-list (py:runresult "arr")
  clear-side "L"
  clear-side "R"
  fill-box "L" preconv
end

to fill-box [side image]
  ;; Formulaicly center inside the left or right box
  let x 0
  ifelse side = "L" [set x 25 clear-side "L"][set x 375 clear-side "R"]
  let width item 1 matrix:dimensions image
  let height item 0 matrix:dimensions image
  let i 0 ;; matrix row i
  let j 0 ;; matrix row j
  while [ i <= (height - 1) ] [
    set j 0
    while [ j <= (width - 1) ] [
      ask patch (j + floor ((200 - width) / 2) + x) ((height - 1 - i) + floor ((200 - height) / 2) + 25) [
        set pcolor (gs-to-rgb (matrix:get image i j))
      ]
      set j (j + 1)
    ]
    set i (i + 1)
  ]
  ifelse side = "L" [set leftbox true][set rightbox true]
end

to clear-side [side]
  if side = "L" [ ask patches with [ ((pxcor >= 25 and pxcor <= 224) and (pycor >= 25 and pycor <= 224)) ] [ set pcolor 49 ] set leftbox false ]
  if side = "R" [ ask patches with [ ((pxcor >= 375 and pxcor <= 574) and (pycor >= 25 and pycor <= 224)) ] [ set pcolor 49 ] set rightbox false ]
end

to-report gs-to-rgb [gs]
  let rdgnbl (list)
  repeat 3 [ set rdgnbl lput gs rdgnbl ]
  report rdgnbl
end

to display-kernel [kernel kernDispX kernDispY]
  ;; Adjust agent font size depending on filter size
  ;ask patches with [ pxcor >= 250 and pxcor <= 349 and pycor >= 50 and pycor <= 149 ] [ ask turtles-here [ die ] ]
  let kHeight item 0 matrix:dimensions kernel
  let kWidth item 1 matrix:dimensions kernel
  let vJump floor (50 / kHeight)
  let hJump floor (50 / kWidth)
  let i 0
  let j 0
  while [ i < kHeight ] [
    set j 0
    while [ j < kWidth ] [
      ask patch (kernDispX + (j * hJump)) (kernDispY - (i * vJump)) [ sprout-kerns 1 [ set label (matrix:get kernel i j) set label-color black set color 69 ] ]
      set j (j + 1)
    ]
    set i (i + 1)
  ]
end

;; TRACER RELATED ;;

to reset-tracer
  ask tracers [ die ]
  create-tracers 1 [
    set size 10
    set color red
    set heading 180
    set xcor 10
    set ycor 10
    set resides "L"
  ]
end

to mousing
  if not (leftbox and rightbox) [
    user-message "There is no image to analyze."
    stop
  ]
  ask tracers with [resides = "L"] [
    setxy (round mouse-xcor) (round mouse-ycor) + 5
  ]
  if [ key? ] of patch (round mouse-xcor) (round mouse-ycor) [
    clear-output
    output-print (word (round mouse-xcor) " " (round mouse-ycor))
    ;; generate the square box with blown up selected area, then figure out the arrow in the rightbox
    if mouse-down? [
      ask zoomies [ die ]
      ask tracers with [resides = "R"] [ die ]
      blow-up
      blow-up2
    ]
  ]
end

to blow-up ;; kernel selection
  let dispX 250
  let dispY 200
  let kernSize 3 ;; Prewitt/Sobel
  if Operator = "Custom" [
    set kernSize length (item 2 read-from-string custom-settings)
  ]
  let squares patch-set patches with [ pxcor <= (round mouse-xcor) + (kernSize - 1) and pxcor >= (round mouse-xcor) and pycor >= (round mouse-ycor) - (kernSize - 1) and pycor <= (round mouse-ycor) ]
  let vJump floor (50 / kernSize)
  let hJump floor (50 / kernSize)
  ask squares [
    sprout-zoomies 1 [
      ifelse pcolor = 49 or pcolor = 69 [ set color black ] [ set color pcolor ]
      set size (hJump)
      set shape "square"
      setxy (dispX + hJump * (pxcor - (round mouse-xcor))) (dispY - vJump * ((round mouse-ycor) - pycor))
    ]
  ]
end

to blow-up2 ;; target patch of rightbox and tracer of rightbox
  let target-patch [points-at] of patch (round mouse-xcor) (round mouse-ycor)
  let x item 0 target-patch
  let y item 1 target-patch
  ask patch 324 184 [
    sprout-zoomies 1 [
      set size 25
      set shape "square"
      set color ([pcolor] of patch x y)
    ]
  ]
  ask patch x y [
    sprout-tracers 1 [
      set size 10
      set heading 180
      set color red
      set resides "R" ;; comment out if you want to keep your R tracers
      setxy xcor (ycor + 5)
    ]
  ]
end

to designate-key-patches
  ask patches [ set key? false ]
  let width (item 1 matrix:dimensions preconv)
  let height (item 0 matrix:dimensions preconv)
  let w (item 1 matrix:dimensions postprocessed)
  let h (item 0 matrix:dimensions postprocessed)
  let maxI height - 1
  let maxJ width - 1
  ;; If operator = sobel, vertical (Prewitt), horizontal (Prewitt)
  let p 1
  let stride 1
  let kernSize 3

  let minX (0 + floor ((200 - width) / 2) + 25) - p
  let maxX (maxJ + floor ((200 - width) / 2) + 25) + p
  let minY ((height - 1 - maxI) + floor ((200 - height) / 2) + 25) - p
  let maxY ((height - 1 - 0) + floor ((200 - height) / 2) + 25) + p

  let currX minX
  let currY maxY
  let currI 0 ;;
  let currJ 0 ;;
  while [currY - (kernSize - 1) >= minY] [
    set currX minX
    set currJ 0
    while [currX + (kernSize - 1) <= maxX] [
      ask patch currX currY [ set key? true set points-at (list (currJ + floor ((200 - w) / 2) + 375) ((h - 1 - currI) + floor ((200 - h) / 2) + 25)) ]
      set currX (currX + stride)
      set currJ (currJ + 1)
    ]
    set currY (currY - stride)
    set currI (currI + 1)
  ]
end

;; CONVOLUTION RELATED ;;

to Vertical ;; Prewitt vertical kernel
  let s 1
  let vpad 1 ;; generate-same-padding
  let hpad 1
  let paddedIm (pad vpad hpad) ;; padding with zeroes
  let kernel matrix:from-row-list [[1 0 -1] [1 0 -1] [1 0 -1]] ;; try flipping
  display-kernel kernel 250 99
  ;stop
  set postconv convolution s paddedIm kernel ;; Convolve
  set postconv matrix:map [ x -> x * (1 / 6) ] postconv
  set postconv threshold
  set postprocessed normalize-gradient
  fill-box "R" postprocessed
end

to Horizontal ;; Prewitt horizontal kernel
  let s 1
  let vpad 1 ;; generate-same-padding
  let hpad 1
  let paddedIm (pad vpad hpad) ;; padding with zeroes
  let kernel matrix:from-row-list [[1 1 1] [0 0 0] [-1 -1 -1]] ;; try flipping
  display-kernel kernel 250 99
  ;stop
  set postconv convolution s paddedIm kernel ;; Convolve
  set postconv matrix:map [ x -> x * (1 / 6) ] postconv
  set postconv threshold
  set postprocessed normalize-gradient
  fill-box "R" postprocessed
end

to Sobel
  let s 1
  let vpad 1
  let hpad 1
  let paddedIm (pad vpad hpad)
  let k_x matrix:from-row-list [[1 0 -1] [2 0 -2] [1 0 -1]]
  let k_y matrix:from-row-list [[1 2 1] [0 0 0] [-1 -2 -1]]
  display-kernel k_x 250 99
  display-kernel k_y 300 99
  let G_x convolution s paddedIm k_x
  set G_x matrix:map [ a -> a * (1 / 8)] G_x
  let G_y convolution s paddedIm k_y
  set G_y matrix:map [ a -> a * (1 / 8) ] G_y
  set postconv matrix:map sqrt (matrix:plus (matrix:map [ a -> a ^ 2] G_x) (matrix:map [ a -> a ^ 2] G_y)) ;; sqrt(G_x^2 + G_y^2)
  print (word "gradient magnitudes: " postconv)
  set postprocessed normalize-gradient
  fill-box "R" postprocessed
end

to Sharpen ;; thres, and thres + norm looked best
  let s 1
  let vpad 1
  let hpad 1
  let paddedIm (pad vpad hpad)
  let kernel matrix:from-row-list [[0 -1 0] [-1 5 -1] [0 -1 0]]
  display-kernel kernel 250 99
  set postconv convolution s paddedIm kernel
  ; set postconv matrix:map [ x -> x * (1 / 9) ] postconv
  set postconv threshold
  set postprocessed normalize-gradient
  fill-box "R" postprocessed
end

to Blur
  let s 1
  let vpad 1
  let hpad 1
  let paddedIm (pad vpad hpad)
  let kernel matrix:from-row-list [[1 1 1] [1 1 1] [1 1 1]]
  display-kernel kernel 250 99
  set postconv convolution s paddedIm kernel
  set postprocessed matrix:map [ x -> x * (1 / 9) ] postconv ;; no need to threshold or multiply by 255/max, everything should be in [0,255]
  fill-box "R" postprocessed
end

to Custom
  let settings read-from-string custom-settings
  let stride item 0 settings
  let padding item 1 settings
  let paddedIm (pad padding padding)
  let kernel matrix:from-row-list (item 2 settings)
  let normFactor sum-matrix (matrix:map abs kernel)
  display-kernel kernel 250 99
  set postconv convolution stride paddedIm kernel
  ; set postconv matrix:map [ x -> x * (1 / normFactor) ] postconv
  ; set postconv threshold
  ; set postprocessed normalize-gradient
  fill-box "R" postprocessed
end

to-report convolution [s paddedIm kernel]
  ;; O_H/O_W = [(HoW + 2P - F) / S] + 1 ;; currently P = 0 because the image is already padded.
  let outHeight floor (((item 0 matrix:dimensions paddedIm) + (2 * 0) - (item 0 matrix:dimensions kernel)) / s) + 1
  let outWidth floor (((item 1 matrix:dimensions paddedIm) + (2 * 0) - (item 1 matrix:dimensions kernel)) / s) + 1
  let kHeight item 0 matrix:dimensions kernel
  let kWidth item 1 matrix:dimensions kernel

  py:set "h" outHeight
  py:set "w" outWidth
  (py:run
   "import numpy as np"
   "empty = np.full((h, w), 0)" ;; np.zeros, change to some other value if you like
  )
  let result matrix:from-row-list (py:runresult "empty")

  let vert 0 ;; counts vertical kernel ops
  let horz 0 ;; counts horizontal kernel ops
  let i 0 ;; anchor top-left
  let j 0 ;; anchor top-left
  while [ vert < outHeight ] [
    set horz 0
    set j 0
    while [ horz < outWidth ] [
      let convSlice (matrix:times-element-wise (matrix:submatrix paddedIm i j (i + kHeight) (j + kWidth)) kernel)
      matrix:set result vert horz (sum-matrix convSlice)
      set horz (horz + 1)
      set j (j + s)
    ]
    set vert (vert + 1)
    set i (i + s)
  ]
  print (word "Right after convolving: " result)
  report result
end
;; consider np.convolve?

to-report normalize-gradient
  ;; find max, multiply by 255/max
  py:set "b" (matrix:to-row-list postconv)
  (py:run
   "import numpy as np"
   "myMax = np.amax(b)"
  )
  let myMax py:runresult "myMax"
  report matrix:map [ x -> x * (255 / myMax) ] postconv
end

to-report threshold
  report (matrix:map [x -> (ifelse-value x < 0 [0] x > 255 [255] [x])] postconv)
end

to-report sum-matrix [convSlice]
  py:set "b" (matrix:to-row-list convSlice)
  (py:run
   "import numpy as np"
   "mySum = np.sum(b)"
  )
  report py:runresult "mySum"
end

to-report pad [vpad hpad]
  ;; Assumes padding with 0
  let unpaddedIm (matrix:to-row-list preconv)
  let width (item 1 matrix:dimensions preconv)
  let paddedIm (list)
  let oneRowVertPad (n-values (width + (2 * hpad)) [0])

  let i 0
  while [ i < length unpaddedIm ] [
    let unpaddedRow item i unpaddedIm
    repeat hpad [
      set unpaddedRow (fput 0 unpaddedRow) ;; stick a 0 to the left
      set unpaddedRow (lput 0 unpaddedRow) ;; stick a 0 to the right
    ]
    ;; Push this now-padded row (list) to the back of the paddedIm list
    set paddedIm (lput unpaddedRow paddedIm)
    set i (i + 1)
  ]

  ;; Now add the top and bottom padding
  repeat vpad [
    set paddedIm (fput oneRowVertPad paddedIm) ;; stick a row of 0s on top
    set paddedIm (lput oneRowVertPad paddedIm) ;; stick a row of 0s on bottom
  ]

  report matrix:from-row-list paddedIm
end

to-report generate-same-padding [h w f s]
  report [ 0 0 ]
end

;; ANCILLARY ;;

to save-image
  if not rightbox [
    user-message "There is no image to save."
    stop
  ]
  let saveLoc user-directory
  py:set "A" (matrix:to-row-list postprocessed)
  py:set "basePath" saveLoc
  py:set "filter" Operator
  (py:run
   "from PIL import Image, ImageOps"
   "import datetime"
   "import numpy as np"
   "im = Image.fromarray(np.array(A))"
   "ritenow = datetime.datetime.now().strftime('%y%m%d_%H%M%S')"
   "im = ImageOps.grayscale(im)"
   "im = im.save(basePath + ritenow + '_' + filter + '.png')"
  )
end
@#$#@#$#@
GRAPHICS-WINDOW
12
12
1040
446
-1
-1
1.7
1
20
1
1
1
0
0
0
1
0
599
0
249
0
0
1
ticks
30.0

BUTTON
16
463
101
497
Setup/Reset
setup
NIL
1
T
OBSERVER
NIL
S
NIL
NIL
1

BUTTON
16
506
101
540
Import Pic
import-pic
NIL
1
T
OBSERVER
NIL
I
NIL
NIL
1

CHOOSER
113
452
205
497
Operator
Operator
"Vertical" "Horizontal" "Sobel" "Sharpen" "Blur" "Custom"
4

BUTTON
217
461
302
494
Convolve
if not leftbox [\n  user-message \"You need to supply an image first.\"\n  stop \n]\nclear-side \"R\"\nask kerns [ die ]\nprofiler:reset\nprofiler:start\nrun Operator\nprofiler:stop\nprint (word \"Time: \" (precision (profiler:inclusive-time Operator / 1000) 3) \"s\")\n\ndesignate-key-patches\nprint \"All done!\"\n
NIL
1
T
OBSERVER
NIL
C
NIL
NIL
1

BUTTON
318
461
382
494
Save!
save-image
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
640
461
720
494
Cam On!
carefully [\n  vid:camera-select\n  vid:start\n] [ user-message error-message ]
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
734
461
816
494
Cam Off!
vid:close
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
686
504
763
537
Capture
carefully [\n  let image (vid:capture-image 500 500)\n  bitmap:export image \"CamPic.png\"\n  import-pic\n] [\n  user-message error-message\n  stop\n]
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

TEXTBOX
113
572
326
599
Running one convolve operation on a 200 x 200 picture takes about 21 seconds.
11
0.0
1

BUTTON
414
461
517
494
Tracer Reset
reset-tracer
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
528
461
605
494
Mousing
mousing
T
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

OUTPUT
414
507
517
536
11

INPUTBOX
112
506
340
566
custom-settings
[1 1 [[0 -1 0] [-1 5 -1] [0 -1 0]]]
1
0
String

@#$#@#$#@
## WHAT IS IT?

(a general understanding of what the model is trying to show or explain)

## HOW IT WORKS

(what rules the agents use to create the overall behavior of the model)

## HOW TO USE IT

(how to use the model, including a description of each of the items in the Interface tab)

## THINGS TO NOTICE

(suggested things for the user to notice while running the model)

## THINGS TO TRY

(suggested things for the user to try to do (move sliders, switches, etc.) with the model)

## EXTENDING THE MODEL

(suggested things to add or change in the Code tab to make the model more complicated, detailed, accurate, etc.)

## NETLOGO FEATURES

(interesting or unusual features of NetLogo that the model uses, particularly in the Code tab; or where workarounds were needed for missing features)

## RELATED MODELS

(models in the NetLogo Models Library and elsewhere which are of related interest)

## CREDITS AND REFERENCES

(a reference to the model's URL on the web if it has one, as well as any other necessary credits, citations, and links)
@#$#@#$#@
default
true
0
Polygon -7500403 true true 150 5 40 250 150 205 260 250

airplane
true
0
Polygon -7500403 true true 150 0 135 15 120 60 120 105 15 165 15 195 120 180 135 240 105 270 120 285 150 270 180 285 210 270 165 240 180 180 285 195 285 165 180 105 180 60 165 15

arrow
true
0
Polygon -7500403 true true 150 0 0 150 105 150 105 293 195 293 195 150 300 150

box
false
0
Polygon -7500403 true true 150 285 285 225 285 75 150 135
Polygon -7500403 true true 150 135 15 75 150 15 285 75
Polygon -7500403 true true 15 75 15 225 150 285 150 135
Line -16777216 false 150 285 150 135
Line -16777216 false 150 135 15 75
Line -16777216 false 150 135 285 75

bug
true
0
Circle -7500403 true true 96 182 108
Circle -7500403 true true 110 127 80
Circle -7500403 true true 110 75 80
Line -7500403 true 150 100 80 30
Line -7500403 true 150 100 220 30

butterfly
true
0
Polygon -7500403 true true 150 165 209 199 225 225 225 255 195 270 165 255 150 240
Polygon -7500403 true true 150 165 89 198 75 225 75 255 105 270 135 255 150 240
Polygon -7500403 true true 139 148 100 105 55 90 25 90 10 105 10 135 25 180 40 195 85 194 139 163
Polygon -7500403 true true 162 150 200 105 245 90 275 90 290 105 290 135 275 180 260 195 215 195 162 165
Polygon -16777216 true false 150 255 135 225 120 150 135 120 150 105 165 120 180 150 165 225
Circle -16777216 true false 135 90 30
Line -16777216 false 150 105 195 60
Line -16777216 false 150 105 105 60

car
false
0
Polygon -7500403 true true 300 180 279 164 261 144 240 135 226 132 213 106 203 84 185 63 159 50 135 50 75 60 0 150 0 165 0 225 300 225 300 180
Circle -16777216 true false 180 180 90
Circle -16777216 true false 30 180 90
Polygon -16777216 true false 162 80 132 78 134 135 209 135 194 105 189 96 180 89
Circle -7500403 true true 47 195 58
Circle -7500403 true true 195 195 58

circle
false
0
Circle -7500403 true true 0 0 300

circle 2
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240

cow
false
0
Polygon -7500403 true true 200 193 197 249 179 249 177 196 166 187 140 189 93 191 78 179 72 211 49 209 48 181 37 149 25 120 25 89 45 72 103 84 179 75 198 76 252 64 272 81 293 103 285 121 255 121 242 118 224 167
Polygon -7500403 true true 73 210 86 251 62 249 48 208
Polygon -7500403 true true 25 114 16 195 9 204 23 213 25 200 39 123

cylinder
false
0
Circle -7500403 true true 0 0 300

dot
false
0
Circle -7500403 true true 90 90 120

face happy
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 255 90 239 62 213 47 191 67 179 90 203 109 218 150 225 192 218 210 203 227 181 251 194 236 217 212 240

face neutral
false
0
Circle -7500403 true true 8 7 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Rectangle -16777216 true false 60 195 240 225

face sad
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 168 90 184 62 210 47 232 67 244 90 220 109 205 150 198 192 205 210 220 227 242 251 229 236 206 212 183

fish
false
0
Polygon -1 true false 44 131 21 87 15 86 0 120 15 150 0 180 13 214 20 212 45 166
Polygon -1 true false 135 195 119 235 95 218 76 210 46 204 60 165
Polygon -1 true false 75 45 83 77 71 103 86 114 166 78 135 60
Polygon -7500403 true true 30 136 151 77 226 81 280 119 292 146 292 160 287 170 270 195 195 210 151 212 30 166
Circle -16777216 true false 215 106 30

flag
false
0
Rectangle -7500403 true true 60 15 75 300
Polygon -7500403 true true 90 150 270 90 90 30
Line -7500403 true 75 135 90 135
Line -7500403 true 75 45 90 45

flower
false
0
Polygon -10899396 true false 135 120 165 165 180 210 180 240 150 300 165 300 195 240 195 195 165 135
Circle -7500403 true true 85 132 38
Circle -7500403 true true 130 147 38
Circle -7500403 true true 192 85 38
Circle -7500403 true true 85 40 38
Circle -7500403 true true 177 40 38
Circle -7500403 true true 177 132 38
Circle -7500403 true true 70 85 38
Circle -7500403 true true 130 25 38
Circle -7500403 true true 96 51 108
Circle -16777216 true false 113 68 74
Polygon -10899396 true false 189 233 219 188 249 173 279 188 234 218
Polygon -10899396 true false 180 255 150 210 105 210 75 240 135 240

house
false
0
Rectangle -7500403 true true 45 120 255 285
Rectangle -16777216 true false 120 210 180 285
Polygon -7500403 true true 15 120 150 15 285 120
Line -16777216 false 30 120 270 120

leaf
false
0
Polygon -7500403 true true 150 210 135 195 120 210 60 210 30 195 60 180 60 165 15 135 30 120 15 105 40 104 45 90 60 90 90 105 105 120 120 120 105 60 120 60 135 30 150 15 165 30 180 60 195 60 180 120 195 120 210 105 240 90 255 90 263 104 285 105 270 120 285 135 240 165 240 180 270 195 240 210 180 210 165 195
Polygon -7500403 true true 135 195 135 240 120 255 105 255 105 285 135 285 165 240 165 195

line
true
0
Line -7500403 true 150 0 150 300

line half
true
0
Line -7500403 true 150 0 150 150

pentagon
false
0
Polygon -7500403 true true 150 15 15 120 60 285 240 285 285 120

person
false
0
Circle -7500403 true true 110 5 80
Polygon -7500403 true true 105 90 120 195 90 285 105 300 135 300 150 225 165 300 195 300 210 285 180 195 195 90
Rectangle -7500403 true true 127 79 172 94
Polygon -7500403 true true 195 90 240 150 225 180 165 105
Polygon -7500403 true true 105 90 60 150 75 180 135 105

plant
false
0
Rectangle -7500403 true true 135 90 165 300
Polygon -7500403 true true 135 255 90 210 45 195 75 255 135 285
Polygon -7500403 true true 165 255 210 210 255 195 225 255 165 285
Polygon -7500403 true true 135 180 90 135 45 120 75 180 135 210
Polygon -7500403 true true 165 180 165 210 225 180 255 120 210 135
Polygon -7500403 true true 135 105 90 60 45 45 75 105 135 135
Polygon -7500403 true true 165 105 165 135 225 105 255 45 210 60
Polygon -7500403 true true 135 90 120 45 150 15 180 45 165 90

sheep
false
15
Circle -1 true true 203 65 88
Circle -1 true true 70 65 162
Circle -1 true true 150 105 120
Polygon -7500403 true false 218 120 240 165 255 165 278 120
Circle -7500403 true false 214 72 67
Rectangle -1 true true 164 223 179 298
Polygon -1 true true 45 285 30 285 30 240 15 195 45 210
Circle -1 true true 3 83 150
Rectangle -1 true true 65 221 80 296
Polygon -1 true true 195 285 210 285 210 240 240 210 195 210
Polygon -7500403 true false 276 85 285 105 302 99 294 83
Polygon -7500403 true false 219 85 210 105 193 99 201 83

square
false
0
Rectangle -7500403 true true 30 30 270 270

square 2
false
0
Rectangle -7500403 true true 30 30 270 270
Rectangle -16777216 true false 60 60 240 240

star
false
0
Polygon -7500403 true true 151 1 185 108 298 108 207 175 242 282 151 216 59 282 94 175 3 108 116 108

target
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240
Circle -7500403 true true 60 60 180
Circle -16777216 true false 90 90 120
Circle -7500403 true true 120 120 60

tracer
false
0
Rectangle -2674135 true false 0 0 300 300

tree
false
0
Circle -7500403 true true 118 3 94
Rectangle -6459832 true false 120 195 180 300
Circle -7500403 true true 65 21 108
Circle -7500403 true true 116 41 127
Circle -7500403 true true 45 90 120
Circle -7500403 true true 104 74 152

triangle
false
0
Polygon -7500403 true true 150 30 15 255 285 255

triangle 2
false
0
Polygon -7500403 true true 150 30 15 255 285 255
Polygon -16777216 true false 151 99 225 223 75 224

truck
false
0
Rectangle -7500403 true true 4 45 195 187
Polygon -7500403 true true 296 193 296 150 259 134 244 104 208 104 207 194
Rectangle -1 true false 195 60 195 105
Polygon -16777216 true false 238 112 252 141 219 141 218 112
Circle -16777216 true false 234 174 42
Rectangle -7500403 true true 181 185 214 194
Circle -16777216 true false 144 174 42
Circle -16777216 true false 24 174 42
Circle -7500403 false true 24 174 42
Circle -7500403 false true 144 174 42
Circle -7500403 false true 234 174 42

turtle
true
0
Polygon -10899396 true false 215 204 240 233 246 254 228 266 215 252 193 210
Polygon -10899396 true false 195 90 225 75 245 75 260 89 269 108 261 124 240 105 225 105 210 105
Polygon -10899396 true false 105 90 75 75 55 75 40 89 31 108 39 124 60 105 75 105 90 105
Polygon -10899396 true false 132 85 134 64 107 51 108 17 150 2 192 18 192 52 169 65 172 87
Polygon -10899396 true false 85 204 60 233 54 254 72 266 85 252 107 210
Polygon -7500403 true true 119 75 179 75 209 101 224 135 220 225 175 261 128 261 81 224 74 135 88 99

wheel
false
0
Circle -7500403 true true 3 3 294
Circle -16777216 true false 30 30 240
Line -7500403 true 150 285 150 15
Line -7500403 true 15 150 285 150
Circle -7500403 true true 120 120 60
Line -7500403 true 216 40 79 269
Line -7500403 true 40 84 269 221
Line -7500403 true 40 216 269 79
Line -7500403 true 84 40 221 269

wolf
false
0
Polygon -16777216 true false 253 133 245 131 245 133
Polygon -7500403 true true 2 194 13 197 30 191 38 193 38 205 20 226 20 257 27 265 38 266 40 260 31 253 31 230 60 206 68 198 75 209 66 228 65 243 82 261 84 268 100 267 103 261 77 239 79 231 100 207 98 196 119 201 143 202 160 195 166 210 172 213 173 238 167 251 160 248 154 265 169 264 178 247 186 240 198 260 200 271 217 271 219 262 207 258 195 230 192 198 210 184 227 164 242 144 259 145 284 151 277 141 293 140 299 134 297 127 273 119 270 105
Polygon -7500403 true true -1 195 14 180 36 166 40 153 53 140 82 131 134 133 159 126 188 115 227 108 236 102 238 98 268 86 269 92 281 87 269 103 269 113

x
false
0
Polygon -7500403 true true 270 75 225 30 30 225 75 270
Polygon -7500403 true true 30 75 75 30 270 225 225 270
@#$#@#$#@
NetLogo 6.1.1
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
default
0.0
-0.2 0 0.0 1.0
0.0 1 1.0 0.0
0.2 0 0.0 1.0
link direction
true
0
Line -7500403 true 150 150 90 180
Line -7500403 true 150 150 210 180
@#$#@#$#@
0
@#$#@#$#@
