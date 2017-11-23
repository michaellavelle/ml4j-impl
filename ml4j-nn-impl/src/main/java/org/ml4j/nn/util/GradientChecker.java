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

package org.ml4j.nn.util;

import org.ml4j.Matrix;
import org.ml4j.nn.CostAndGradients;
import org.ml4j.nn.FeedForwardNeuralNetworkBase;
import org.ml4j.nn.FeedForwardNeuralNetworkContext;
import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.TrainableAxons;
import org.ml4j.nn.layers.FeedForwardLayer;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.supervised.SupervisedFeedForwardNeuralNetworkImpl;
import org.ml4j.nn.synapses.DirectedSynapses;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * 
 * @author Michael Lavelle
 *
 * @param <C> The type of context used for this neural network.
 * @param <N> The type of neural network.
 */
public class GradientChecker {

  private static final Logger LOGGER = 
      LoggerFactory.getLogger(FeedForwardNeuralNetworkBase.class);
  
  private SupervisedFeedForwardNeuralNetworkImpl feedForwardNeuralNetwork;
  
  /**
   * Contract a gradient checker for this neural network.
   * 
   * @param feedForwardNeuralNetwork The neural network.
   */
  public GradientChecker(SupervisedFeedForwardNeuralNetworkImpl feedForwardNeuralNetwork) {
    this.feedForwardNeuralNetwork = feedForwardNeuralNetwork;
  }

  private List<TrainableAxons<?, ?, ?>> getTrainableAxonsList(
      SupervisedFeedForwardNeuralNetworkImpl feedForwardNeuralNetwork) {

    List<TrainableAxons<?, ?, ?>> trainableAxonsList = new ArrayList<>();
    for (int layerIndex = 0; layerIndex < feedForwardNeuralNetwork
        .getNumberOfLayers(); layerIndex++) {

      FeedForwardLayer<?, ?> layer = feedForwardNeuralNetwork.getLayer(layerIndex);
      for (DirectedSynapses<?, ?> synapses : layer.getSynapses()) {
        Axons<?, ?, ?> axons = synapses.getAxons();
        if (axons != null && axons instanceof TrainableAxons) {
          TrainableAxons<?, ?, ?> trainableAxons = (TrainableAxons<?, ?, ?>) axons;
          trainableAxonsList.add(trainableAxons);
        }
      }
    }
    return trainableAxonsList;

  }

  private double norm(List<Double> data) {
    double total = 0;
    for (int i = 0; i < data.size(); i++) {
      total = total + data.get(i) * data.get(i);
    }
    return Math.sqrt(total);
  }

  /**
   * Performs numerical gradient checks...outputting a metric which represents the difference 
   * between the numerical gradients and the gradient from back prop.
   * 
   * @param inputActivations The input activations.
   * @param desiredOutputActivations The target activations.
   * @param trainingContext The training context
   * @param costAndGradients The cost and gradients from back prop.
   */
  public void checkGradients(NeuronsActivation inputActivations,
      NeuronsActivation desiredOutputActivations, FeedForwardNeuralNetworkContext trainingContext,
      CostAndGradients costAndGradients) {

    LOGGER.info("Performing gradient check.....");
    
    List<Matrix> averageAxonsGradients = costAndGradients.getAverageTrainableAxonsGradients();

    List<TrainableAxons<?, ?, ?>> axonsList = getTrainableAxonsList(feedForwardNeuralNetwork);

    SupervisedFeedForwardNeuralNetworkImpl firstDup = 
        (SupervisedFeedForwardNeuralNetworkImpl)feedForwardNeuralNetwork.dup();

    SupervisedFeedForwardNeuralNetworkImpl secondDup = 
        (SupervisedFeedForwardNeuralNetworkImpl)feedForwardNeuralNetwork.dup();

    for (int axonsIndex = 0; axonsIndex < averageAxonsGradients.size(); axonsIndex++) {

      List<Double> dthetaList = new ArrayList<Double>();
      List<Double> dthetaApproxList = new ArrayList<Double>();
      List<Double> diffList = new ArrayList<Double>();

      Axons<?, ?, ?> axons = axonsList.get(axonsIndex);

      // Transpose the axon gradients into matrices that correspond to the orientation of the
      // connection weights ( COLUMNS_SPAN_FEATURE_SET )
      Matrix axonsGrad = averageAxonsGradients.get(axonsIndex).transpose();

      Matrix axonsConnectionWeights = axons.getConnectionWeights();

      if (true) {

        double epsilon = Math.pow(10, -7);

        // Create two clones of these Axon weights - 
        // we will perturb the weight each of the weights in each clone in turn.
        Axons<?, ?, ?> axonsClone1 = getTrainableAxonsList(firstDup).get(axonsIndex);
        Axons<?, ?, ?> axonsClone2 = getTrainableAxonsList(secondDup).get(axonsIndex);

        // Loop through all the weights in these Axons.
        for (int row = 0; row < axonsConnectionWeights.getRows(); row++) {
          for (int col = 0; col < axonsConnectionWeights.getColumns(); col++) {

            // Adjust the value of this weight by subtracting epsilon in the clone1Weights.
            Matrix clone1Weights = axonsClone1.getConnectionWeights();
            double dtheta = axonsGrad.get(row, col);
            dthetaList.add(dtheta);
            double originalFirstWeightValue = clone1Weights.get(row, col);
            clone1Weights.put(row, col, originalFirstWeightValue - epsilon);

            // Adjust the value of this weight by adding epsilon in the clone2Weights.
            Matrix clone2Weights = axonsClone2.getConnectionWeights();
            double originalSecondWeightValue = clone2Weights.get(row, col);
            clone2Weights.put(row, col, originalSecondWeightValue + epsilon);

            // Get the cost of the clone 1 weights.
            double clone1Cost = firstDup.getAverageCost(inputActivations,
                desiredOutputActivations, trainingContext);

            // Get the cost of the clone 2 weights.
            double clone2Cost = secondDup.getAverageCost(inputActivations,
                desiredOutputActivations, trainingContext);

            // Approximate the gradient of the weight in each of the clones using 
            // the costs for the perturbed weight clones.
            double dthetaApprox = clone2Cost - clone1Cost;
            dthetaApprox = dthetaApprox / (2d * epsilon);
            LOGGER.trace("Numerical gradient:" + dthetaApprox);
            LOGGER.trace("Back prop gradient:" + dtheta);

            dthetaApproxList.add(dthetaApprox);
            double diff = dthetaApprox - dtheta;

            diffList.add(diff);
            System.out.println(axonsIndex + "," + row + "," + col + "," + diff);

            // Reset the values of the weights.
            clone1Weights.put(row, col, originalFirstWeightValue);

            clone2Weights.put(row, col, originalSecondWeightValue);
          }
        }

        double top = norm(diffList);
        double bottom = norm(dthetaApproxList) + norm(dthetaList);
        double result = top / bottom;
        LOGGER.info("Gradient check metric:" + result);
      }
    }
  }
}
