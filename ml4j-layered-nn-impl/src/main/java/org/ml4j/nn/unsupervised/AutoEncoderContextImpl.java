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

package org.ml4j.nn.unsupervised;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.supervised.LayeredFeedForwardNeuralNetworkContextImpl;

/**
 * Simple default implementation of AutoEncoderContext.
 * 
 * @author Michael Lavelle
 * 
 */
public class AutoEncoderContextImpl 
    extends LayeredFeedForwardNeuralNetworkContextImpl implements AutoEncoderContext {

  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;

  public AutoEncoderContextImpl(MatrixFactory matrixFactory, int startLayerIndex,
      Integer endLayerIndex, boolean isTrainingContext) {
    super(matrixFactory, startLayerIndex, endLayerIndex, isTrainingContext);
  }
}
