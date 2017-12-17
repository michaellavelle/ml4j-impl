/*
 * Copyright 2017 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */

package org.ml4j.nn;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.activationfunctions.LinearActivationFunction;
import org.ml4j.nn.activationfunctions.SigmoidActivationFunction;
import org.ml4j.nn.activationfunctions.SoftmaxActivationFunction;
import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.ConnectionWeightsAdjustmentDirection;
import org.ml4j.nn.axons.TrainableAxons;
import org.ml4j.nn.costfunctions.CostFunction;
import org.ml4j.nn.costfunctions.CostFunctionGradient;
import org.ml4j.nn.costfunctions.CrossEntropyCostFunction;
import org.ml4j.nn.costfunctions.DeltaRuleCostFunctionGradientImpl;
import org.ml4j.nn.costfunctions.MultiClassCrossEntropyCostFunction;
import org.ml4j.nn.costfunctions.SumSquaredErrorCostFunction;
import org.ml4j.nn.layers.DirectedLayerActivation;
import org.ml4j.nn.layers.DirectedLayerContext;
import org.ml4j.nn.layers.DirectedLayerGradient;
import org.ml4j.nn.layers.FeedForwardLayer;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.synapses.DirectedSynapses;
import org.ml4j.nn.synapses.DirectedSynapsesContext;
import org.ml4j.nn.synapses.DirectedSynapsesGradient;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Default base implementation of a FeedForwardNeuralNetwork.
 *
 * @author Michael Lavelle
 */
public abstract class FeedForwardNeuralNetworkBase<C extends FeedForwardNeuralNetworkContext, 
    N extends FeedForwardNeuralNetwork<C,N>> 
    implements FeedForwardNeuralNetwork<C, N> {

  private static final Logger LOGGER = 
      LoggerFactory.getLogger(FeedForwardNeuralNetworkBase.class);
  
  private List<FeedForwardLayer<?, ?>> layers;
  
  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;
  
  /**
   * Constructor for a multi-layer FeedForwardNeuralNetwork.
   * 
   * @param layers The layers
   */
  public FeedForwardNeuralNetworkBase(FeedForwardLayer<?, ?>... layers) {
    this.layers = new ArrayList<FeedForwardLayer<?, ?>>();
    this.layers.addAll(Arrays.asList(layers));
  }

  protected void train(NeuronsActivation trainingDataActivations, 
      NeuronsActivation trainingLabelActivations, C trainingContext) {

    final int numberOfEpochs = trainingContext.getTrainingEpochs();
        
    LOGGER.info("Training the FeedForwardNeuralNetwork for "
          + numberOfEpochs + " epochs");
    
    CostAndGradients costAndGradients = null;
    
    for (int i = 0; i < numberOfEpochs; i++) {
      
      if (trainingContext.getTrainingMiniBatchSize() == null) {
        costAndGradients = getCostAndGradients(trainingDataActivations, 
            trainingLabelActivations, trainingContext);
        
        LOGGER.info("Epoch:" + i + " Cost:" + costAndGradients.getAverageCost());
        
        adjustConnectionWeights(trainingContext, 
            costAndGradients.getAverageTrainableAxonsGradients());
      } else {
        int miniBatchSize = trainingContext.getTrainingMiniBatchSize();
        int numberOfTrainingElements = trainingDataActivations.getActivations().getRows();
        int numberOfBatches = (numberOfTrainingElements - 1) / miniBatchSize + 1;
        for (int batchIndex = 0; batchIndex < numberOfBatches; batchIndex++) {
          int startRowIndex = batchIndex * miniBatchSize;
          int endRowIndex =
              Math.min(startRowIndex + miniBatchSize - 1, numberOfTrainingElements - 1);
          int[] rowIndexes = new int[endRowIndex - startRowIndex + 1];
          for (int r = startRowIndex; r <= endRowIndex; r++) {
            rowIndexes[r - startRowIndex] = r;
          }

          Matrix dataBatch = trainingDataActivations.getActivations().getRows(rowIndexes);
          Matrix labelBatch = trainingLabelActivations.getActivations().getRows(rowIndexes);
         
          NeuronsActivation batchDataActivations =
              new NeuronsActivation(dataBatch,
                  trainingDataActivations.getFeatureOrientation());

          NeuronsActivation batchLabelActivations =
              new NeuronsActivation(labelBatch,
                  trainingLabelActivations.getFeatureOrientation());

          costAndGradients = getCostAndGradients(batchDataActivations, 
              batchLabelActivations, trainingContext);
          
          LOGGER.trace("Epoch:" + i + " batch " + batchIndex 
              + " Cost:" + costAndGradients.getAverageCost());
          
          adjustConnectionWeights(trainingContext, 
              costAndGradients.getAverageTrainableAxonsGradients());
          
          batchIndex++;
          
        }
        
        LOGGER.info("Epoch:" + i + " Cost:" + costAndGradients.getAverageCost());

      }
    }
  }
 
  protected CostAndGradients getCostAndGradients(NeuronsActivation inputActivations,
      NeuronsActivation desiredOutputActivations, C trainingContext) {
       
    final CostFunction costFunction = getCostFunction(trainingContext.getMatrixFactory());
    
    // Forward propagate the trainingDataActivations through the entire Network
    ForwardPropagation forwardPropagation =
        forwardPropagate(inputActivations, trainingContext);
    
    CostFunctionGradient costFunctionGradient = 
        new DeltaRuleCostFunctionGradientImpl(costFunction, desiredOutputActivations, 
            forwardPropagation.getOutputs());
   
    // Back propagate the cost function gradient through the network
    BackPropagation backPropagation = forwardPropagation.backPropagate(costFunctionGradient, 
        trainingContext);

    // Obtain the gradients of each set of Axons we wish to train - for this example it is
    // all the Axons
    List<Matrix> totalTrainableAxonsGradients = new ArrayList<>();
    List<DirectedLayerGradient> reversed = new ArrayList<>();
    reversed.addAll(backPropagation.getDirectedLayerGradients());
    Collections.reverse(reversed);

    for (DirectedLayerGradient gradient : reversed) {
      for (DirectedSynapsesGradient synapsesGradient : gradient.getSynapsesGradients()) {
        
        Matrix totalTrainableAxonsGradient = synapsesGradient.getTotalTrainableAxonsGradient();
        
        if (totalTrainableAxonsGradient != null) {
          totalTrainableAxonsGradients.add(totalTrainableAxonsGradient);
        }
      }
    }

    // Obtain the cost from the cost function
    LOGGER.debug("Calculating total cost function cost");
    double totalCost = costFunction.getTotalCost(
        desiredOutputActivations.getActivations(),
        forwardPropagation.getOutputs().getActivations());
    
    double totalRegularisationCost = forwardPropagation.getTotalRegularisationCost(trainingContext);
        
    double totalCostWithRegularisation = totalCost + totalRegularisationCost;
    
    Collections.reverse(totalTrainableAxonsGradients);

    int numberOfTrainingExamples = inputActivations.getActivations().getRows();
    
    return new CostAndGradients(totalCostWithRegularisation, 
          totalTrainableAxonsGradients, numberOfTrainingExamples);
    
  }
  
  private List<TrainableAxons<?, ?, ?>> getTrainableAxonsList(C context) {
    
    List<TrainableAxons<?, ?, ?>> trainableAxonsList = new ArrayList<>();
    for (int layerIndex = 0; layerIndex < getNumberOfLayers(); layerIndex++) {
      DirectedLayerContext layerContext = context.getLayerContext(layerIndex);
      FeedForwardLayer<?, ?> layer = getLayer(layerIndex);
      int synapsesIndex = 0;
      for (DirectedSynapses<?, ?> synapses : layer.getSynapses()) {
        DirectedSynapsesContext synapsesContext = 
            layerContext.getSynapsesContext(synapsesIndex);
        Axons<?, ? , ?> axons = synapses.getAxons();
        if (axons != null && axons.isTrainable(synapsesContext.getAxonsContext(0))) {
          TrainableAxons<?, ?, ?> trainableAxons = 
              (TrainableAxons<?, ?, ?>) axons;
          trainableAxonsList.add(trainableAxons);
        }
        synapsesIndex++;
      }
    }
    return trainableAxonsList;
    
  }
  
  private void adjustConnectionWeights(C trainingContext, List<Matrix> trainableAxonsGradients) {
    double learningRate = trainingContext.getTrainingLearningRate();
    
    List<TrainableAxons<?, ?, ?>> trainableAxonsList = getTrainableAxonsList(trainingContext);
    
    for (int axonsIndex = 0; axonsIndex < trainableAxonsGradients.size(); axonsIndex++) {
      TrainableAxons<?, ?, ?> trainableAxons = trainableAxonsList.get(axonsIndex);
      // Transpose the axon gradients into matrices that correspond to the orientation of the
      // connection weights ( COLUMNS_SPAN_FEATURE_SET )
      Matrix axonsGrad = trainableAxonsGradients.get(axonsIndex).transpose();

      // Adjust the weights of each set of Axons by subtracting the learning-rate scaled
      // gradient matrices
      trainableAxons.adjustConnectionWeights(axonsGrad.mul(learningRate), 
          ConnectionWeightsAdjustmentDirection.SUBTRACTION);
    }
  }

  @Override
  public List<FeedForwardLayer<?, ?>> getLayers() {
    return layers;
  }

  @Override
  public int getNumberOfLayers() {
    return layers.size();
  }

  @Override
  public FeedForwardLayer<?, ?> getLayer(int layerIndex) {
    return layers.get(layerIndex);
  }

  @Override
  public FeedForwardLayer<?, ?> getFirstLayer() {
    return layers.get(0);
  }

  @Override
  public FeedForwardLayer<?, ?> getFinalLayer() {
    return layers.get(getNumberOfLayers() - 1);

  }

  @Override
  public ForwardPropagation forwardPropagate(NeuronsActivation inputActivation,
      FeedForwardNeuralNetworkContext context) {
    
    int endLayerIndex =
        context.getEndLayerIndex() == null ? (getNumberOfLayers() - 1) : context.getEndLayerIndex();

    LOGGER.debug("Forward propagating through FeedForwardNeuralNetwork from layerIndex:"
        + context.getStartLayerIndex() + " to layerIndex:" + endLayerIndex);
        
    NeuronsActivation inFlightActivations = inputActivation;
    int layerIndex = 0;
    List<DirectedLayerActivation> activations = new ArrayList<>();
    for (FeedForwardLayer<?, ?> layer : getLayers()) {

      if (layerIndex >= context.getStartLayerIndex() && layerIndex <= endLayerIndex) {

        DirectedLayerActivation inFlightLayerActivations = 
            layer.forwardPropagate(inFlightActivations, context.getLayerContext(layerIndex));
        activations.add(inFlightLayerActivations);
        inFlightActivations = inFlightLayerActivations.getOutput();
      }
      layerIndex++;
    }
    
    return new ForwardPropagationImpl(activations, inFlightActivations);
  }
  
  /**
   * @return The default cost function for use by this Network.
   */
  protected CostFunction getCostFunction(MatrixFactory matrixFactory) {

    List<DirectedSynapses<?, ?>> synapseList = getFinalLayer().getSynapses();
    DirectedSynapses<?, ?> finalSynapses = synapseList.get(synapseList.size() - 1);
    DifferentiableActivationFunction activationFunction = finalSynapses.getActivationFunction();
    if (activationFunction == null) {
      throw new UnsupportedOperationException(
          "Default cost function not yet defined for null activation function");
    }
    if (activationFunction instanceof SigmoidActivationFunction) {
      LOGGER.debug("Defaulting to use CrossEntropyCostFunction");
      return new CrossEntropyCostFunction();
    } else if (activationFunction instanceof SoftmaxActivationFunction) {
      LOGGER.debug("Defaulting to use MultiClassCrossEntropyCostFunction");
      return new MultiClassCrossEntropyCostFunction();
    } else if (activationFunction instanceof LinearActivationFunction) {
      LOGGER.debug("Defaulting to use SumSquredErrorCostFunction");
      return new SumSquaredErrorCostFunction();
    } else {
      throw new UnsupportedOperationException(
          "Default cost function not yet defined for:" + activationFunction.getClass());
    }
  }
}
