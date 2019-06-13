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

package org.ml4j.nn.supervised;

import java.util.List;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.CostAndGradientsImpl;
import org.ml4j.nn.ForwardPropagation;
import org.ml4j.nn.LayeredFeedForwardNeuralNetworkBase;
import org.ml4j.nn.LayeredFeedForwardNeuralNetworkContext;
import org.ml4j.nn.layers.DirectedLayerChain;
import org.ml4j.nn.layers.DirectedLayerChainImpl;
import org.ml4j.nn.layers.FeedForwardLayer;
import org.ml4j.nn.neurons.NeuronsActivation;

/**
 * Default implementation of SupervisedFeedForwardNeuralNetwork.
 *
 * @author Michael Lavelle
 */
public class LayeredSupervisedFeedForwardNeuralNetworkImpl 
    extends LayeredFeedForwardNeuralNetworkBase<LayeredFeedForwardNeuralNetworkContext,
    LayeredSupervisedFeedForwardNeuralNetwork> implements LayeredSupervisedFeedForwardNeuralNetwork {
  
  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;
  
  
  public LayeredSupervisedFeedForwardNeuralNetworkImpl(
		DirectedLayerChain<FeedForwardLayer<?, ?>> initialisingComponentChain) {
	super(initialisingComponentChain);
  }
  
  public LayeredSupervisedFeedForwardNeuralNetworkImpl(List<FeedForwardLayer<?, ?>> layers) {
		super(new DirectedLayerChainImpl<>(layers));
	  }

  @Override
  public void train(NeuronsActivation trainingDataActivations,
      NeuronsActivation trainingLabelActivations, LayeredFeedForwardNeuralNetworkContext trainingContext) {
    super.train(trainingDataActivations, trainingLabelActivations, trainingContext);
  }
  
  /**
   * Return the prediction accuracy.
   * 
   * @param inputActivations The input activations.
   * @param desiredClassificationActivations The desired prediction activations.
   * @param context The context.
   * @return The accuracy
   */
  @Override
  public double getClassificationAccuracy(NeuronsActivation inputActivations,
      NeuronsActivation desiredClassificationActivations, LayeredFeedForwardNeuralNetworkContext context) {

    // Forward propagate the trainingDataActivations
    ForwardPropagation forwardPropagation = forwardPropagate(inputActivations, context);

    Matrix predictions = getClassifications(forwardPropagation.getOutputs().getActivations(),
        context.getMatrixFactory());

    return computeAccuracy(predictions, desiredClassificationActivations.getActivations());
  }

  private Matrix getClassifications(Matrix outputActivations, MatrixFactory matrixFactory) {

    Matrix predictions = 
        matrixFactory.createZeros(outputActivations.getRows(), outputActivations.getColumns());
    for (int row = 0; row < outputActivations.getRows(); row++) {

      int index = outputActivations.getRow(row).argmax();
      predictions.put(row, index, 1);
    }
    return predictions;
  }

  /**
   * Helper function to compute the accuracy of predictions using calculated predictions predictions
   * and correct output matrix.
   *
   * @param predictions The predictions
   * @param Y The desired output labels
   * @return The accuracy of the network
   */
  protected double computeAccuracy(Matrix predictions, Matrix outputs) {
    return ((predictions.mul(outputs)).sum()) * 100 / outputs.getRows();
  }

  @Override
  public LayeredSupervisedFeedForwardNeuralNetwork dup() {
	  throw new UnsupportedOperationException("Not yet implemented");
    //return new SupervisedFeedForwardNeuralNetworkImpl2(getLayer(0), getLayer(1));
  }
  
  @Override
  public CostAndGradientsImpl getCostAndGradients(NeuronsActivation inputActivations,
      NeuronsActivation desiredOutpuActivations, LayeredFeedForwardNeuralNetworkContext trainingContext) {
    return super.getCostAndGradients(inputActivations, desiredOutpuActivations, trainingContext);
  }
}
