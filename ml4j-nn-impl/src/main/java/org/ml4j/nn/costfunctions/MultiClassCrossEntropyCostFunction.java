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

package org.ml4j.nn.costfunctions;

import org.ml4j.Matrix;

/**
 * Multi class cross entropy cost function.
 * 
 * @author Michael Lavelle
 *
 */
public class MultiClassCrossEntropyCostFunction implements CostFunction {

  @Override
  public double getTotalCost(Matrix desiredOutputs, Matrix actualOutputs) {
   
    Matrix jpart = (desiredOutputs.mul(-1).mul(limitLog(actualOutputs))).rowSums();

    return jpart.sum();
  }

  private double limit(double value) {
    value = Math.min(value, 1 - 0.000000000000001);
    value = Math.max(value, 0.000000000000001);
    return value;
  }

  private Matrix limitLog(Matrix matrix) {
    Matrix dupMatrix = matrix.dup();
    for (int i = 0; i < dupMatrix.getLength(); i++) {
      dupMatrix.put(i, (double) Math.log(limit(dupMatrix.get(i))));
    }
    return dupMatrix;
  }

  @Override
  public double getAverageCost(Matrix desiredOutputs, Matrix actualOutputs) {
    return getTotalCost(desiredOutputs, actualOutputs) / desiredOutputs.getRows();
  }
}
