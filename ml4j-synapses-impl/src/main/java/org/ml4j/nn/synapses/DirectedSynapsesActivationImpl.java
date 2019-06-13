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

package org.ml4j.nn.synapses;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.ml4j.Matrix;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunctionActivation;
import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.AxonsActivation;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.axons.AxonsGradient;
import org.ml4j.nn.axons.AxonsGradientImpl;
import org.ml4j.nn.axons.DirectedAxonsComponentActivation;
import org.ml4j.nn.axons.TrainableAxons;
import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.components.DirectedComponentGradientImpl;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.costfunctions.CostFunctionGradient;
import org.ml4j.nn.graph.DirectedDipoleGraph;
import org.ml4j.nn.graph.DirectedPath;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Default implementation of DirectedSynapsesActivation.
 * 
 * @author Michael Lavelle
 */
public class DirectedSynapsesActivationImpl extends DirectedSynapsesActivationBase {

  private static final Logger LOGGER =
      LoggerFactory.getLogger(DirectedSynapsesActivationImpl.class);

  /**
   * 
   * @param synapses The synapses.
   * @param inputActivation The input activation.
   * @param axonsActivation The axons activation.
   * @param activationFunctionActivation The activation function activation.
   * @param outputActivation The output activation.
   */
  public DirectedSynapsesActivationImpl(DirectedSynapses<?, ?> synapses,
      NeuronsActivation inputActivation, DirectedDipoleGraph<DirectedAxonsComponentActivation> axonsActivation,
      DifferentiableActivationFunctionActivation activationFunctionActivation,
      NeuronsActivation outputActivation, DirectedComponentsContext synapsesContext) {
    super(synapses, inputActivation, axonsActivation, activationFunctionActivation,
        outputActivation, synapsesContext);
  }


  @Override
  public DirectedComponentGradient<NeuronsActivation> backPropagate(DirectedComponentGradient<NeuronsActivation> da) {

    LOGGER.debug("Back propagating through synapses activation....");

    validateAxonsAndAxonsActivation();
    
    DirectedComponentGradient<NeuronsActivation> dz = activationFunctionActivation.backPropagate(da);

    return backPropagateThroughAxons(dz, synapsesContext);
  }


  @Override
  public DirectedComponentGradient<NeuronsActivation> backPropagate(CostFunctionGradient da) {

    LOGGER.debug("Back propagating through synapses activation....");

    validateAxonsAndAxonsActivation();

    DirectedComponentGradient<NeuronsActivation> dz = activationFunctionActivation.backPropagate(da);

    return backPropagateThroughAxons(dz, this.synapsesContext);
  }

  
  private void validateAxonsAndAxonsActivation() {

    if (synapses.getRightNeurons().hasBiasUnit()) {
      throw new IllegalStateException(
          "Backpropagation through axons with a rhs bias unit not supported");
    }

    if (axonsActivationGraph == null || axonsActivationGraph.getParallelPaths().isEmpty()) {
      throw new IllegalStateException(
          "The synapses activation is expected to contain an AxonsActivation path");
    } else {

      boolean allEmpty = true;
      for (DirectedPath<DirectedAxonsComponentActivation> parallelPath : axonsActivationGraph.getParallelPaths()) {
        if (parallelPath != null && !parallelPath.getEdges().isEmpty()) {
          allEmpty = false;
        }
      }
      if (allEmpty) {
        throw new IllegalStateException(
            "The synapses activation is expected to contain an AxonsActivation");
      }
    }
      
  }

  private DirectedComponentGradient<NeuronsActivation> backPropagateThroughAxons(DirectedComponentGradient<NeuronsActivation> dz,
		  DirectedComponentsContext synapsesContext) {

    LOGGER.debug("Pushing data right to left through axons...");

    List<AxonsGradient> totalTrainableAxonsGradients = new ArrayList<>();
    totalTrainableAxonsGradients.addAll(dz.getTotalTrainableAxonsGradients());

    DirectedComponentGradient<NeuronsActivation> inputGradient = dz;
    int pathIndex = 0;
    for (DirectedPath<DirectedAxonsComponentActivation> parallelAxonsPath : axonsActivationGraph
        .getParallelPaths()) {

      DirectedComponentGradient<NeuronsActivation> gradientToBackPropagate = dz;

      int axonsIndex = parallelAxonsPath.getEdges().size() - 1;

      List<DirectedAxonsComponentActivation> reversedAxonsActivations = new ArrayList<DirectedAxonsComponentActivation>();
      reversedAxonsActivations.addAll(parallelAxonsPath.getEdges());
      Collections.reverse(reversedAxonsActivations);
      
      for (DirectedAxonsComponentActivation axonsActivation : reversedAxonsActivations) {

        //AxonsContext axonsContext = synapsesContext.getAxonsContext(pathIndex, axonsIndex);
        
        
        inputGradient = axonsActivation.backPropagate(gradientToBackPropagate);
        totalTrainableAxonsGradients.addAll(inputGradient.getTotalTrainableAxonsGradients());
       // AxonsGradient totalTrainableAxonsGradient =
         //   getTrainableAxonsGradient(axonsActivation, axonsContext, dz.getOutput());
        //if (totalTrainableAxonsGradient != null) {
          //totalTrainableAxonsGradients.add(totalTrainableAxonsGradient);
        //}

        gradientToBackPropagate = inputGradient;

        axonsIndex++;
      }
      pathIndex--;
    }
    
    return new DirectedComponentGradientImpl<>(totalTrainableAxonsGradients, inputGradient.getOutput());
  }
  
  private AxonsGradient getTrainableAxonsGradient(AxonsActivation axonsActivation, 
      AxonsContext axonsContext, 
      NeuronsActivation gradientToBackPropagate) {
    Axons<?, ?, ?> axons = axonsActivation.getAxons();
    Matrix totalTrainableAxonsGradientMatrix = null;
    AxonsGradient totalTrainableAxonsGradient = null;

    if (axons instanceof TrainableAxons<?, ?, ?> && axons.isTrainable(axonsContext)) {

      LOGGER.debug("Calculating Axons Gradients");

      totalTrainableAxonsGradientMatrix = gradientToBackPropagate.getActivations()
          .mmul(axonsActivation.getPostDropoutInputWithPossibleBias().getActivationsWithBias());

      if (axonsContext.getRegularisationLambda() != 0) {

        LOGGER.debug("Calculating total regularisation Gradients");

        Matrix connectionWeightsCopy = axons.getDetachedConnectionWeights();

        Matrix firstRow = totalTrainableAxonsGradientMatrix.getRow(0);
        Matrix firstColumn = totalTrainableAxonsGradientMatrix.getColumn(0);

        totalTrainableAxonsGradientMatrix =
            totalTrainableAxonsGradientMatrix.addi(connectionWeightsCopy.muli(
                axonsContext.getRegularisationLambda()));

        if (axons.getLeftNeurons().hasBiasUnit()) {

          totalTrainableAxonsGradientMatrix.putRow(0, firstRow);
        }
        if (axons.getRightNeurons().hasBiasUnit()) {

          totalTrainableAxonsGradientMatrix.putColumn(0, firstColumn);
        }
      }
    }
    if (totalTrainableAxonsGradientMatrix != null) {
      totalTrainableAxonsGradient = new AxonsGradientImpl((TrainableAxons<?, ?, ?>) axons, 
          totalTrainableAxonsGradientMatrix);
    }
    
    return totalTrainableAxonsGradient;
  }
}
