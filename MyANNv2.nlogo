extensions [csv matrix py profiler]

links-own [
  weight         ;; Weight given to end1 activation by end2
  inlayer        ;; Layer index of end2
]

breed [bias-nodes bias-node]
breed [input-nodes input-node]
breed [output-nodes output-node]
breed [hidden-nodes hidden-node]
breed [halos halo]

turtles-own [
  activation
  layer ;; what layer of the NN am I in? (input is 0, hidden/output is 1,2,...,)
  id ;; which neuron of my layer am I? (bias is 0, regular neurons are 1,2,...,)
]

globals [
  traindata
  trainlabels ;; one-hot encoded [0,1,2,...,8,9]
  testdata
  testlabels ;; one-hot encoded
  input-size ;; 400
  output-size ;; 10
  all-layer-sizes ;; in order of: input, hidden, output
  W ;; list of weight matrices (different size) [W1, W2, ..]
  b ;; list of bias matrices (different size) [b1, b2, ...]

  currentEpoch
  currentBatch ;; forward prop
  totalBatches
  batchCost
  Z ;; list of pre-activation values [Z1, Z2, Z3]
  A ;; list of activation values [A1, A2, A3] ** A3 = softmax(Z3), not ReLU

  dW ;; used in backprop
  db
  dZ
  dA

  confidences
  predictions
  truelabels
  results
  metrics
]

to load-data
  clear-all
  file-close-all
  ;; File strings as inputted by the user
  ;; 5000 data from Yan Le Cun's MNIST set, data has already been normalized/shuffled
  let X_trainFile "X_train.csv" ;; user-input "Training data file: " ;; 4000x400
  let Y_trainFile "Y_train.csv" ;; user-input "Training labels file: " ;; 10x4000
  let X_testFile "X_test.csv" ;; user-input "Test data file: " ;; 1000x400
  let Y_testFile "Y_test.csv" ;; user-input "Test labels file: " ;; 10x1000
  ;; When m is the first dimension, NetLogo can load these values into matrices quicker

  ;; Extract data from files, save into local vars
  file-open X_trainFile
  let X_train (list)
  while [not file-at-end?] [
    set X_train lput (csv:from-row file-read-line) X_train
  ]
  file-close

  file-open Y_trainFile
  let Y_train (list)
  while [not file-at-end?] [
    set Y_train lput (csv:from-row file-read-line) Y_train
  ]
  file-close

  file-open X_testFile
  let X_test (list)
  while [not file-at-end?] [
    set X_test lput (csv:from-row file-read-line) X_test
  ]
  file-close

  file-open Y_testFile
  let Y_test (list)
  while [not file-at-end?] [
    set Y_test lput (csv:from-row file-read-line) Y_test
  ]
  file-close

  file-close-all

  ;; At this point each list item is a row for one picture, so when we create our
  ;; matrices, stack each row as a column, horizontally (input size x m training examples)

  ;; 400/100 examples per digit for training/testing
  set traindata matrix:from-column-list X_train ;; matrix:dimensions [400 4000]
  set trainlabels matrix:from-row-list Y_train ;; matrix:dimensions [10 4000]
  set testdata matrix:from-column-list X_test ;; matrix:dimensions [400 1000]
  set testlabels matrix:from-row-list Y_test ;; matrix:dimensions [10 1000]

  set input-size item 0 matrix:dimensions testdata ;; 400
  set output-size item 0 matrix:dimensions testlabels ;; 10
  ;; running this function takes about 9 seconds
end

to setup
  resize-world 0 300 0 150 ;; adjust this
  set-default-shape bias-nodes "face happy"
  set-default-shape input-nodes "dot"
  set-default-shape hidden-nodes "circle"
  set-default-shape output-nodes "square"
  set-default-shape halos "thin ring"

  let hidden-layer-sizes read-from-string hidden-layer-dimensions ;; hidden layer sizes is a list
  set hidden-layer-sizes fput input-size hidden-layer-sizes
  set hidden-layer-sizes lput output-size hidden-layer-sizes
  set all-layer-sizes hidden-layer-sizes

  print all-layer-sizes
  clear-turtles ;; can load-data once and keep reconfiguring the net before training/testing
  init-weights
  setup-nodes
  setup-links
  retrieve-weights
  color-links
  label-nodes
  set currentEpoch 1
  reset-ticks
end

to init-weights
  ;; He/ReLU, Xavier/tanh, random
  set W (list)
  set b (list)

  let l-index 0
  py:setup "C:/Users/Matt/anaconda3/python"
  py:set "scalingFactor" scalingFactor
  py:set "biasInit" bias-init
  let seeding "np.random.seed(3)" ;; change seed # for different random weights
  let weight-setup-string (ifelse-value
    weight-init-method = "He" [ "W = np.random.randn(layerNext, layerCurr) * np.sqrt(2/layerCurr)" ]
    weight-init-method = "Xavier" [ "W = np.random.randn(layerNext, layerCurr) * np.sqrt(1/layerCurr)" ]
    weight-init-method = "random" [ "W = np.random.randn(layerNext, layerCurr) * scalingFactor" ])

  repeat (length all-layer-sizes) - 1 [
    py:set "layerCurr" item l-index all-layer-sizes
    py:set "layerNext" item (l-index + 1) all-layer-sizes
    (py:run
      "import numpy as np"
      seeding
      weight-setup-string
      "b = np.ones((layerNext,1)) * biasInit"
    )
    let currW matrix:from-row-list py:runresult "W"
    let currb matrix:from-row-list py:runresult "b"
    set W lput currW W
    set b lput currb b
    set l-index l-index + 1
  ]
  print word "W: " W
  print word "b: " b
end
;; Borrowed heavily from Jacob Samson's Deep Neural Network and Dropout==============================================
to setup-nodes
  ;; do gradient and go by max height, width
  let segments length all-layer-sizes
  let hidden-sizes sublist all-layer-sizes 1 ((length all-layer-sizes) - 1)

  let l-index 0
  let index 0

  ;; Layer = 0, id = 0, ;;activation = 1
  create-bias-nodes 1 [
    setxy nodex l-index nodey l-index index (input-size + 1)
    set layer l-index
    ;;set activation 1
    set id 0

    set color 65 ; green
  ]

  ; Layer = 0, id = 1...400
  set index 1
  repeat input-size [
    create-input-nodes 1 [
      setxy nodex l-index nodey l-index index (input-size + 1)
      ;;set activation ((random 2) * 2) - 1
      set layer l-index ;; 0
      set id index
      set color 48 ;; yellow
    ]
    set index index + 1
  ]

  set l-index 1
  set index 0

  foreach hidden-sizes [ ?1 ->
    ;; Layer = 1,2, id = 0, ;;activation = 1
    create-bias-nodes 1 [
      setxy nodex l-index nodey l-index index (?1 + 1)
      set layer l-index
      ;;set activation 1
      set id 0
      set color 65 ; green
    ]

    set index 1

    ;; Layer = 1,2, id = 1...10, 1...5
    repeat ?1 [
      create-hidden-nodes 1 [
        setxy nodex l-index nodey l-index index (?1 + 1)
        ;;set activation ((random 2) * 2) - 1
        set layer l-index
        set id index
        set color 27 ;; orange
      ]
     set index index + 1
    ]

    set l-index l-index + 1
    set index 0
  ]

  ;; Layer = 3, id = 1...10, activation = 0
  repeat output-size [
    create-output-nodes 1 [
      setxy nodex l-index nodey l-index index output-size
      ;;set activation ((random 2) * 2) - 1
      set activation 0
      set layer l-index
      set id index + 1
      set color 87 ;; blue
    ]
    set index index + 1
  ]
end

;; Find the appropriate x coordinate for this layer
to-report nodex [l-index]
  report min-pxcor + (((l-index + 1) * (world-width - 1)) / (length all-layer-sizes - 2 + 3))
end

;; Find the appropriate y cooridinate for this node
to-report nodey [l-index index in-layer]
  report max-pycor - (((index + 1) * (world-height - 1)) / (in-layer + 1))
end

to setup-links
  let l-index 0
  repeat ((length all-layer-sizes) - 2) [
   connect-all (turtles with [layer = l-index]) (hidden-nodes with [layer = (l-index + 1)]) ;; bias/input to hidden, bias/hidden to next hidden
   set l-index l-index + 1
  ]
  connect-all (turtles with [layer = l-index]) (output-nodes with [layer = (l-index + 1)]) ;; last bias/hidden to output
end

to connect-all [nodes1 nodes2]
  ask nodes1 [
    create-links-to nodes2 [
      set inlayer [layer] of one-of nodes2
    ]
  ]
end

to color-links
  let l-index 1
  let maxw 0
  repeat (length all-layer-sizes) - 1 [
   set maxw max [abs weight] of links with [inlayer = l-index]
   ask links with [inlayer = l-index] [
     let wquotient (weight / maxw)
     let colorstr (wquotient * 127)
     let colorvec (list (colorstr + 127) (127 - (abs colorstr)) (127 - colorstr) 196)
     set color colorvec
   ]
   set l-index l-index + 1
  ]
end
;; ================end of borrowing=============================
to color-nodes
  ask halos [die]
  let l-index 1
  repeat length A [
    let maxA max [abs activation] of (turtle-set hidden-nodes output-nodes) with [layer = l-index]
    ask (turtle-set hidden-nodes output-nodes) with [layer = l-index] [
      let transparency 0
      ifelse maxA = 0
        [ set transparency 0 ]
        [ set transparency ((abs activation) / maxA * 255) ]
      hatch-halos 1 [
        set size 2
        __set-line-thickness 1
        ifelse activation > 0
          [ set color lput transparency [255 255 0] ]
          [ set color lput transparency [0 255 0] ]
        set label ""
      ]
    ]
    set l-index l-index + 1
  ]
end

to retrieve-weights
  let l-index 1
  repeat length all-layer-sizes - 1 [
    let W-matrix item (l-index - 1) W
    let b-matrix item (l-index - 1) b
    ask links with [inlayer = l-index] [
      ifelse [id] of end1 > 0  ;; connections from input or hidden nodes
        [ set weight matrix:get W-matrix (([id] of end2) - 1) (([id] of end1) - 1) ]
        ;; connections from a bias node
        [ set weight matrix:get b-matrix (([id] of end2) - 1) 0 ]
    ]
    set l-index l-index + 1
  ]
end

to retrieve-activations
  let l-index 1
  repeat length A [
    ask (turtle-set hidden-nodes output-nodes) with [layer = l-index] [
      set activation matrix:get (item (l-index - 1) A) (id - 1) 0 ;; first datum in minibatch is chosen
    ]
    set l-index l-index + 1
  ]
end

to label-nodes
  if node-labels? [
    ask (turtle-set bias-nodes hidden-nodes output-nodes) [
      set label (word layer ", " id "    ") ;; for testing purposes
    ]
    ask (input-nodes) [
      if (id mod 10 = 0) [
        set label (word layer ", " id "    ") ;; for testing purposes
      ]
    ]
  ]
end

to run-through
  if currentEpoch > numEpochs [
    ; user-message (word "All training data has been run through for " numEpochs " epochs.")
    stop
  ]
  set currentBatch 1 ;; to initialize for training thru the whole dataset
  set totalBatches ceiling ((item 1 matrix:dimensions traindata) / minibatchSize) ;; 4000 / 40 = 100

  set-current-plot "Batch Error"
  set-plot-x-range 0 totalBatches

  print (word "Epoch #: " currentEpoch)
  repeat totalBatches [ train ] ; 1 epoch

  set-current-plot "Batch Error"
  clear-plot
  set-current-plot "Epoch Error"
  plotxy currentEpoch batchCost

  set currentEpoch currentEpoch + 1
end

to train
  print (word "Batch #: " currentBatch)
  set Z (list)
  set A (list)
  ; set totalBatches ceiling ((item 1 matrix:dimensions traindata) / minibatchSize) ;; 4000 / 40 = 100
  if currentBatch > totalBatches [
    ; user-message "All training data has been seen by the net."
    stop
  ]
  ;; for epoch 2, minibatchSize = 40, lb = 40, rb = 79
  let leftbound (currentBatch - 1) * minibatchSize
  let rightbound (ifelse-value
    minibatchSize = 1 [leftbound] ;; stochastic gradient descent
    currentBatch = totalBatches [(item 1 matrix:dimensions traindata) - 1] ;; batch gradient descent
                               [currentBatch * minibatchSize - 1]) ;; minibatch gradient descent
  ; print (word leftbound ", " rightbound)

  ;; Carve out the minibatch we want: rowstart (in) colstart (in) rowend (ex) colend (ex)
  let minibatch matrix:submatrix traindata 0 leftbound (item 0 matrix:dimensions traindata) (rightbound + 1)

  ;; Forward prop
  forward-propagation minibatch
  ; print item (length Z - 1) Z
  ; print item (length A - 1) A
  retrieve-activations
  color-nodes

  ;; Calculate cost, plot
  let outputs (last A)
  let labels matrix:submatrix trainlabels 0 leftbound (item 0 matrix:dimensions trainlabels) (rightbound + 1)
  set batchCost cost-fxn outputs labels
  tick
  set-current-plot "Batch Error"
  plotxy currentBatch batchCost

  ;; Back prop
  backward-propagation minibatch labels

  ;; Recolor links
  retrieve-weights
  color-links

  set currentBatch currentBatch + 1
end

to forward-propagation [minibatch]
  ;; 400, 10, 5, 10. Input is 400 x 40 (one minibatch)
  let batchSize item 1 matrix:dimensions minibatch
  let W-index 0
  let currA minibatch ;; minibatch = input = A0
  repeat (length W) - 1 [ ;; get Z1, A1 and Z2, A2
    let currZ matrix:plus (matrix:times (item W-index W) currA) (broadcast (item W-index b) batchSize) ;; Z = W*A + b
    set Z lput currZ Z
    set currA (runresult (word "matrix:map " activation-fxn " currZ"))
    set A lput currA A
    set W-index W-index + 1
  ]
  let currZ matrix:plus (matrix:times (item W-index W) currA) (broadcast (item W-index b) batchSize)
  set Z lput currZ Z
  ;; softmax activate
  set currA softmax currZ
  set A lput currA A
end

to backward-propagation [minibatch labels] ;; need minibatch for A0, labels for Y (as in A_L - Y)
  set dW (list)
  set db (list)
  set dZ (list)
  set dA (list)
  let l-index length W ;; 3 in a 400, 10, 5, 10 network
  let batchSize item 1 matrix:dimensions minibatch
  ;; Put an item 0 in so that list index matches the layer index (no need to +/- 1)
  let W_ fput -1 W ;; [-1 W1 W2 W3]
  let b_ fput -1 b ;; [-1 b1 b2 b3]
  let Z_ fput -1 Z ;; [-1 Z1 Z2 Z3]
  let A_ fput minibatch A ;; [minibatch A1 A2 A3]
  ;; Calculate gradients
  while [l-index != 0] [
    if l-index != length W [
      set dA fput (matrix:times (matrix:transpose (item (l-index + 1) W_)) (first dZ)) dA ;; dA[L-1] = W[L]T dot dZ[L]
    ]
    ifelse l-index = length W
    [set dZ fput (matrix:minus (last A_) labels) dZ] ;; dZ[L] = A[L] - Y ;; initiate backprop
    [set dZ fput (matrix:times-element-wise (first dA) (runresult (word "matrix:map " activation-fxn "-deriv (item l-index Z_)"))) dZ] ;; dZ[L] = dA[L] * g'(Z[L])
    set dW fput (matrix:times (1 / batchSize) (matrix:times (first dZ) (matrix:transpose (item (l-index - 1) A_)))) dW ;; 1/m * dZ[L] dot A[L-1]'
    set db fput (matrix:times (1 / batchSize) (special-sum (first dZ))) db ;; 1/m * np.sum(dZ[L], axis = 1, keepDims = True)
    set l-index l-index - 1
  ]
  ;; Update weights, no need to do it in a separate fxn
  set l-index (length W - 1)
  while [l-index >= 0] [
    set W replace-item l-index W (matrix:minus (item l-index W) (matrix:times learning-rate (item l-index dW)))
    set b replace-item l-index b (matrix:minus (item l-index b) (matrix:times learning-rate (item l-index db)))
    set l-index l-index - 1
  ]
end

to predict ;; should look very similar to forward propagation
  set confidences (list)
  set predictions (list)
  set truelabels (list)
  set results 0
  set metrics 0
  let testSize item 1 matrix:dimensions testlabels ;; testdata would work as well
  let W-index 0
  let matrix testdata
  repeat (length W) - 1 [
    set matrix matrix:plus (matrix:times (item W-index W) matrix) (broadcast (item W-index b) testSize) ;; Z = W*A + b
    set matrix (runresult (word "matrix:map " activation-fxn " matrix"))
    set W-index W-index + 1
  ]
  set matrix matrix:plus (matrix:times (item W-index W) matrix) (broadcast (item W-index b) testSize) ;; final Z = W*A + b
  set matrix softmax matrix
  ; print matrix:to-column-list matrix
  ;; Hardmax in case some of the softmax outputs are very close to 1 but not 1
  set matrix hardmax matrix
  ; print matrix:to-column-list matrix
  set predictions un1hot-encode matrix
  set truelabels un1hot-encode testlabels
  present-results predictions truelabels ;; both 1000-element lists
end

to present-results [netLabels realLabels]
  print (word " confidences: " confidences)
  print (word "net predicts: " netLabels)
  print (word "the real num: " realLabels)
  let correct 0
  let table matrix:from-row-list [[0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0] [0 0 0 0 0 0 0 0 0 0]]
  ;; col 0 = true +, col 1 = false +, col 2 = true -, col 3 = false -
  (foreach netLabels realLabels [ [net real] ->
    if net = real [set correct correct + 1]
    matrix:set table net real ((matrix:get table net real) + 1)
  ])
  set results table

  let metric-pretty (list)
  set metrics (word "======Epoch: " (currentEpoch - 1) "================================\n\n")
  set metrics (word metrics "Digit | Preczn | Recall | F1 Score \n")
  let digit 0
  let beta 1 ;; recall is (beta) times as important as precision. tune to calc F1 differently
  repeat 10 [
    let precizion -1
    let recall -1
    let f1score -1
    let p-denom (reduce + (matrix:get-row table digit))
    let r-denom (reduce + (matrix:get-column table digit))
    if p-denom != 0 [set precizion precision ((matrix:get table digit digit) / p-denom) 3]
    if r-denom != 0 [set recall precision ((matrix:get table digit digit) / r-denom) 3]
    let f-denom (((beta ^ 2) * precizion) + recall)
    if (f-denom != 0) [set f1score precision ((1 + (beta ^ 2)) * (precizion * recall) / f-denom) 3]
    set metrics (word metrics "    " digit " \t" precizion "\t " recall "\t  " f1score "\n")
    set metric-pretty lput (list digit precizion recall f1score) metric-pretty
    set digit digit + 1
  ]

  let examples length realLabels
  let accuracy correct / examples
  ;let accuracy calc-accuracy table
  ;let precizion calc-precision table
  ;let recall calc-recall table
  set metrics (word metrics "\nOverall accuracy: " (precision (accuracy * 100) 2) "%\n")
  print metrics

  file-open "log.txt"
  file-print date-and-time
  file-print (word "======Epoch: " (currentEpoch - 1) "================================")
  file-print (word "Layer sizes: " all-layer-sizes ", Weight Init: " weight-init-method ", scalingFactor: " scalingFactor ", activation: " activation-fxn ", minibatchSize: " minibatchSize ", learning rate: " learning-rate ", bias-init: " bias-init)
  file-print ""
  file-print "[Digit Preczn Recall F1 Score]"
  foreach metric-pretty [ x -> file-print x ]
  file-print ""
  file-print (word "Overall accuracy: " (precision (accuracy * 100) 2) "% =================")
  file-print ""
  file-close
  ;; print matrix:to-column-list table
end

to-report calc-accuracy [table]
  let truePos reduce + (matrix:get-column table 0)
  let falsePos reduce + (matrix:get-column table 1)
  let trueNeg reduce + (matrix:get-column table 2)
  let falseNeg reduce + (matrix:get-column table 3)
  report (truePos + trueNeg) / (truePos + trueNeg + falsePos + falseNeg)
end

to-report calc-precision [table]
  ;; col 0 = true +, col 1 = false +, col 2 = true -, col 3 = false -
  let truePos reduce + (matrix:get-column table 0)
  let falsePos reduce + (matrix:get-column table 1)
  report (truePos / (truePos + falsePos))
end

to-report calc-recall [table]
  ;; col 0 = true +, col 1 = false +, col 2 = true -, col 3 = false -
  let truePos reduce + (matrix:get-column table 0)
  let falseNeg reduce + (matrix:get-column table 3)
  report (truePos / (truePos + falseNeg))
end

to-report special-sum [mat]
  let matList matrix:to-row-list mat
  let dZ_sum (list)
  foreach matList
  [ x -> set dZ_sum lput (lput (reduce + x) (list)) dZ_sum ]
  set dZ_sum matrix:from-row-list dZ_sum
  report dZ_sum
end

to-report cost-fxn [outputs labels]
  ; print outputs
  ; print labels
  let outputCols matrix:to-column-list outputs
  let labelCols matrix:to-column-list labels
  let batchSize length labelCols
  let cost 0
  let datum-index 0
  repeat batchSize [
    let outputCol item datum-index outputCols ;; a single list
    let labelCol item datum-index labelCols ;; a single list
    (foreach outputCol labelCol
      [ [y-hat y] -> set cost cost + (y * ln y-hat) ])
    set datum-index datum-index + 1
  ]
  report cost / batchSize * -1
end

to-report un1hot-encode [mat]
  let un1hot (list)
  let colList matrix:to-column-list mat
  foreach colList [
    x -> set un1hot lput (position 1 x) un1hot
  ]
  report un1hot
end

to-report hardmax [mat]
  let colList matrix:to-column-list mat
  let hardMatrix (list)
  foreach colList [
    x -> set hardMatrix lput (hmax-helper x) hardMatrix
  ]
  report matrix:from-column-list hardMatrix
end

to-report hmax-helper [col] ; a single list
  let maximum max col
  (ifelse
    maximum >= 0.75 [ set confidences lput "S" confidences ]
    maximum >= 0.50 [ set confidences lput "G" confidences ]
    [ set confidences lput "M" confidences ])
  report map [ i -> floor (i / maximum) ] col ;; if i is the max value you get 1, and if not you get 0
end

to-report softmax [mat]
  let colList matrix:to-column-list mat
  let softMatrix (list)
  foreach colList [
    x -> set softMatrix lput (smax-helper x) softMatrix
  ]
  report matrix:from-column-list softMatrix
end

to-report smax-helper [col] ; a single list
  let esum 0
  let softCol (list)
  foreach col [
    x -> set esum esum + exp x
  ]
  foreach col [
    x -> set softCol lput ((exp x) / esum) softCol
  ]
  report softCol
end

to-report broadcast [biasMatrix expansion]
  let biasCol item 0 (matrix:to-column-list biasMatrix)
  let broadcastedBias n-values expansion [biasCol]
  set broadcastedBias matrix:from-column-list broadcastedBias
  report broadcastedBias
end

to-report tanh [input]
  report ((e ^ input) - (e ^ (-1 * input))) / ((e ^ input) + (e ^ (-1 * input)))
end

to-report tanh-deriv [input]
  report 1 - ((tanh input) ^ 2)
end

to-report sigmoid [input]
  report 1 / (1 + (e ^ (-1 * input)))
end

to-report sigmoid-deriv [input]
  report (sigmoid input) * (1 - sigmoid input)
end

to-report ReLU [input]
  report max list 0 input
end

to-report ReLU-deriv [input]
  ifelse (input > 0) [report 1][report 0]
end

to-report swish [input]
  report input * sigmoid input
end

to-report swish-deriv [input]
  report (swish input) + ((sigmoid input) * (1 - (swish input)))
end

to export
  export-view "exports/View.png"
  export-interface "exports/Interface.png"
  export-all-plots "exports/Batch_And_Epoch_Errors.csv"
  let i 0
  let bigString ""
  while [i < length W] [
    set bigString (word bigString (word (csv:to-string matrix:to-row-list item i W) "\n\n" (csv:to-string matrix:to-row-list item i b) "\n\n"))
    set i (i + 1)
  ]
  file-open "Weights.csv"
  file-type bigString
  file-close
end
@#$#@#$#@
GRAPHICS-WINDOW
250
10
1579
681
-1
-1
4.39
1
10
1
1
1
0
0
0
1
0
300
0
150
0
0
1
ticks
30.0

BUTTON
7
28
88
61
load-data
profiler:reset\nprofiler:start\nload-data\nprofiler:stop\nprint (word \"Loaded data in: \" (precision (profiler:inclusive-time \"load-data\" / 1000) 3) \"s\")
NIL
1
T
OBSERVER
NIL
L
NIL
NIL
1

TEXTBOX
8
10
158
28
1. Load training/test data
11
0.0
1

INPUTBOX
6
87
139
147
hidden-layer-dimensions
[20 10 10]
1
0
String

TEXTBOX
6
69
156
87
2. Configure the net
11
0.0
1

BUTTON
34
210
97
243
setup
profiler:reset\nprofiler:start\nsetup\nprofiler:stop\nprint (word \"Network set up: \" (precision (profiler:inclusive-time \"setup\" / 1000) 3) \"s\")
NIL
1
T
OBSERVER
NIL
S
NIL
NIL
1

SWITCH
116
210
241
243
node-labels?
node-labels?
0
1
-1000

CHOOSER
5
155
143
200
weight-init-method
weight-init-method
"He" "Xavier" "random"
0

INPUTBOX
155
139
240
199
scalingFactor
10.0
1
0
Number

TEXTBOX
7
250
157
268
3. Training settings
11
0.0
1

CHOOSER
6
265
144
310
activation-fxn
activation-fxn
"ReLU" "tanh" "sigmoid" "swish"
1

BUTTON
14
386
78
419
trainOnce
profiler:reset\nprofiler:start\nrun-through\nprofiler:stop\nprint (word \"Epoch time: \" (precision (profiler:inclusive-time \"run-through\" / 1000) 3) \"s\")
NIL
1
T
OBSERVER
NIL
T
NIL
NIL
1

INPUTBOX
152
251
240
311
minibatchSize
40.0
1
0
Number

BUTTON
89
386
148
419
fullyTrain
profiler:reset\nprofiler:start\nrun-through\nprofiler:stop\nprint (word \"Epoch time: \" (precision (profiler:inclusive-time \"run-through\" / 1000) 3) \"s\")
T
1
T
OBSERVER
NIL
Y
NIL
NIL
1

PLOT
1385
84
1803
302
Batch Error
Batch
Error
0.0
400.0
0.0
0.5
true
false
"" ""
PENS
"default" 1.0 0 -16777216 true "" ""

INPUTBOX
152
316
241
376
learning-rate
0.1
1
0
Number

BUTTON
160
386
230
419
predict
profiler:reset\nprofiler:start\npredict\nprofiler:stop\nprint (word \"Predictions: \" (precision (profiler:inclusive-time \"predict\" / 1000) 3) \"s\")
NIL
1
T
OBSERVER
NIL
P
NIL
NIL
1

INPUTBOX
31
316
120
376
numEpochs
5.0
1
0
Number

PLOT
1385
329
1656
498
Epoch Error
Epoch
Error
0.0
10.0
0.0
0.5
true
false
"" ""
PENS
"default" 1.0 0 -16777216 true "" ""

TEXTBOX
105
30
234
58
Data determines size of input and output
11
0.0
1

INPUTBOX
154
73
239
133
bias-init
0.0
1
0
Number

BUTTON
1683
397
1752
430
Export
export
NIL
1
T
OBSERVER
NIL
E
NIL
NIL
1

@#$#@#$#@
## WHAT IS IT?

(a general understanding of what the model is trying to show or explain)
Written in NetLogo 6.1.1

## HOW IT WORKS

(what rules the agents use to create the overall behavior of the model)

## HOW TO USE IT

(how to use the model, including a description of each of the items in the Interface tab)

Using Base Anaconda environment, add to system user variables...

## THINGS TO NOTICE

(suggested things for the user to notice while running the model)

## THINGS TO TRY

(suggested things for the user to try to do (move sliders, switches, etc.) with the model)


## EXTENDING THE MODEL

(suggested things to add or change in the Code tab to make the model more complicated, detailed, accurate, etc.)

## NETLOGO FEATURES

(interesting or unusual features of NetLogo that the model uses, particularly in the Code tab; or where workarounds were needed for missing features)

## RELATED MODELS

--> Jacob Samson's Deep Neural Networks and Dropout in model commons, Uri Wilensky's model in model commons

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

thin ring
true
0
Circle -7500403 false true -1 -1 301

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
<experiments>
  <experiment name="He batch GD" repetitions="1" runMetricsEveryStep="true">
    <setup>load-data
setup</setup>
    <go>run-through</go>
    <final>predict</final>
    <enumeratedValueSet variable="scalingFactor">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="weight-init-method">
      <value value="&quot;He&quot;"/>
      <value value="&quot;Xavier&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="hidden-layer-dimensions">
      <value value="&quot;[80]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="activation-fxn">
      <value value="&quot;swish&quot;"/>
      <value value="&quot;tanh&quot;"/>
      <value value="&quot;ReLU&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="node-labels?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="minibatchSize">
      <value value="4000"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="numEpochs">
      <value value="10"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="learning-rate">
      <value value="0.05"/>
      <value value="0.1"/>
      <value value="0.15"/>
      <value value="0.2"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="bias-init">
      <value value="0.01"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="random weight init" repetitions="1" runMetricsEveryStep="true">
    <setup>load-data
setup</setup>
    <go>run-through</go>
    <final>predict</final>
    <enumeratedValueSet variable="weight-init-method">
      <value value="&quot;random&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="scalingFactor">
      <value value="0.001"/>
      <value value="0.01"/>
      <value value="1"/>
      <value value="10"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="hidden-layer-dimensions">
      <value value="&quot;[20 10 10]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="activation-fxn">
      <value value="&quot;ReLU&quot;"/>
      <value value="&quot;swish&quot;"/>
      <value value="&quot;tanh&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="numEpochs">
      <value value="10"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="node-labels?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="minibatchSize">
      <value value="10"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="learning-rate">
      <value value="0.005"/>
      <value value="0.01"/>
      <value value="0.03"/>
      <value value="0.05"/>
      <value value="0.1"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="400 40 10, with bias init" repetitions="1" runMetricsEveryStep="true">
    <setup>load-data
setup</setup>
    <go>run-through</go>
    <final>predict</final>
    <enumeratedValueSet variable="weight-init-method">
      <value value="&quot;He&quot;"/>
      <value value="&quot;Xavier&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="scalingFactor">
      <value value="10"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="hidden-layer-dimensions">
      <value value="&quot;[80]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="activation-fxn">
      <value value="&quot;swish&quot;"/>
      <value value="&quot;tanh&quot;"/>
      <value value="&quot;ReLU&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="numEpochs">
      <value value="10"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="node-labels?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="minibatchSize">
      <value value="1"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="learning-rate">
      <value value="0.01"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="bias-init">
      <value value="0.01"/>
      <value value="0.1"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="experiment" repetitions="1" runMetricsEveryStep="true">
    <setup>load-data
setup</setup>
    <go>run-through</go>
    <final>predict</final>
    <enumeratedValueSet variable="weight-init-method">
      <value value="&quot;He&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="scalingFactor">
      <value value="10"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="hidden-layer-dimensions">
      <value value="&quot;[20 10 10]&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="activation-fxn">
      <value value="&quot;tanh&quot;"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="bias-init">
      <value value="0"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="numEpochs">
      <value value="5"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="node-labels?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="minibatchSize">
      <value value="10"/>
      <value value="20"/>
      <value value="40"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="learning-rate">
      <value value="0.08"/>
      <value value="0.1"/>
      <value value="0.12"/>
    </enumeratedValueSet>
  </experiment>
</experiments>
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
