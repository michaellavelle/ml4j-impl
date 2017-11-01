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

package org.ml4j.nn.activationfunctions;

import org.ml4j.Matrix;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The default Linear Activation Function.
 * 
 * @author Michael Lavelle
 *
 */
public class LinearActivationFunction implements DifferentiableActivationFunction {

  private static final Logger LOGGER = LoggerFactory.getLogger(LinearActivationFunction.class);

  @Override
  public NeuronsActivation activate(NeuronsActivation input, NeuronsActivationContext context) {
    LOGGER.debug("Activating through LinearActivationFunction");
    if (input.isBiasUnitIncluded()) {
      throw new UnsupportedOperationException(
          "Activations passing through activation function should not include a bias unit"
          + " as this has not yet been implemented");
    }
    return new NeuronsActivation(input.getActivations().dup(), input.isBiasUnitIncluded(),
        input.getFeatureOrientation());
  }

  @Override
  public NeuronsActivation activationGradient(NeuronsActivation outputActivation,
      NeuronsActivationContext context) {
    
    if (outputActivation.isBiasUnitIncluded()) {
      throw new UnsupportedOperationException(
          "Activations passing through activation function should not include a bias unit"
          + " as this has not yet been implemented");
    }
    
    LOGGER.debug("Performing linear gradient of NeuronsActivation");
    
    Matrix gradientMatrix = context.getMatrixFactory()
        .createOnes(outputActivation.getActivations().getRows(), 
        outputActivation.getActivations().getColumns());
    
    return new NeuronsActivation(gradientMatrix,
        outputActivation.isBiasUnitIncluded(), outputActivation.getFeatureOrientation());

  }
}