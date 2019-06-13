/*
 * Copyright 2017 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.ml4j.nn;

import org.ml4j.nn.components.TrailingActivationFunctionDirectedComponentChainActivation;
import org.ml4j.nn.costfunctions.CostFunctionGradient;
import org.ml4j.nn.neurons.NeuronsActivation;

/**
 * Default implementation of ForwardPropagation.
 * 
 * @author Michael Lavelle
 */
public class ForwardPropagationImpl implements ForwardPropagation {
  
  private TrailingActivationFunctionDirectedComponentChainActivation activationChain;
  
  /**
   * Create a new ForwardPropagation from the output activations at the
   * right hand side of a DirectedNeuralNetwork after a forward propagation.
   * 
   * @param activations All the DirectedLayerActivation instaces generated
   *        by the forward propagation.
   * @param outputActivations The output activations at the
   *        right hand side of a DirectedNeuralNetwork after a forward propagation.
   */
  public ForwardPropagationImpl(TrailingActivationFunctionDirectedComponentChainActivation activationChain) {
    super();
    this.activationChain = activationChain;
  }

  @Override
  public NeuronsActivation getOutputs() {
    return activationChain.getOutput();
  }

  @Override
  public BackPropagation backPropagate(CostFunctionGradient neuronActivationGradients, 
      DirectedNeuralNetworkContext context) {
    BackPropagation backPropagation =  new BackPropagationImpl(activationChain.backPropagate(neuronActivationGradients));
    if (context.getBackPropagationListener() != null) {
      context.getBackPropagationListener().onBackPropagation(backPropagation);
    }
    return backPropagation;
  }

  //@Override
  //public double getAverageRegularisationCost(DirectedNeuralNetworkContext context) {
    //return getTotalRegularisationCost(context) / getOutputs().getActivations().getRows();
  //}

  @Override
  public double getTotalRegularisationCost(DirectedNeuralNetworkContext context) {
   double totalRegularisationCost = 0d;
    //int layerIndex = 0;
    /*
    for (ChainableDirectedComponentActivation<NeuronsActivation> activation : activationChain.getActivations()) {
      totalRegularisationCost = totalRegularisationCost + activation.getTotalRegularisationCost(
          context.getLayerContext(layerIndex++));
    }
    */
    return totalRegularisationCost;
  }
  
  /*
  @Override
  public List<DirectedLayerActivation> getLayerActivations() {
    return activationChain.getActivations();
  }
  */
 
}
