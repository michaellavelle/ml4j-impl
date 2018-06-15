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

package org.ml4j.nn.axons;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.NeuronsActivationWithPossibleBiasUnit;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Default base Axons implementation.
 * 
 * @author Michael Lavelle
 *
 * @param <L> The type of Neurons on the left hand side of these Axons
 * @param <R> The type of Neurons on the right hand side of these Axons
 * @param <A> The type of these Axons
 */
public abstract class AxonsBase<L extends Neurons, R extends Neurons, 
    A extends Axons<L, R, A>, C extends AxonsConfig>
    implements Axons<L, R, A> {

  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;

  private static final Logger LOGGER = LoggerFactory.getLogger(AxonsBase.class);

  protected L leftNeurons;
  protected R rightNeurons;
  protected Matrix connectionWeights;
  protected C config;
  protected ConnectionWeightsMask connectionWeightsMask;

  /**
   * Construct a new Axons instance.
   * 
   * @param leftNeurons The Neurons on the left hand side of these Axons
   * @param rightNeurons The Neurons on the right hand side of these Axons
   * @param connectionWeights The connection weights.
   * @param connectionWeightsMask The connection weights mask.
   * @param config The config for these Axons.
   */
  public AxonsBase(L leftNeurons, R rightNeurons, MatrixFactory matrixFactory,
      Matrix connectionWeights, ConnectionWeightsMask connectionWeightsMask, C config) {
    this(leftNeurons, rightNeurons,
        matrixFactory.createZeros(leftNeurons.getNeuronCountIncludingBias(),
            rightNeurons.getNeuronCountIncludingBias()),
        connectionWeightsMask, config);
    adjustConnectionWeights(connectionWeights, ConnectionWeightsAdjustmentDirection.ADDITION, true);
  }

  /**
   * Construct a new AxonsBase instance.
   * 
   * @param leftNeurons The Neurons on the left hand side of these Axons
   * @param rightNeurons The Neurons on the right hand side of these Axons
   * @param matrixFactory The matrix factory.
   * @param config The config for these Axons.
   */
  public AxonsBase(L leftNeurons, R rightNeurons, MatrixFactory matrixFactory, C config) {
    this.config = config;
    this.leftNeurons = leftNeurons;
    this.rightNeurons = rightNeurons;
    this.connectionWeightsMask = createConnectionWeightsMask(matrixFactory);
    this.connectionWeights = matrixFactory.createZeros(leftNeurons.getNeuronCountIncludingBias(),
        rightNeurons.getNeuronCountIncludingBias());
    validateConnectionWeightsAndConnectionWeightsMaskDimensions();
    adjustConnectionWeights(createDefaultInitialConnectionWeights(matrixFactory),
        ConnectionWeightsAdjustmentDirection.ADDITION, true);
  }

  /**
   * Construct a new AxonsBase instance.
   * 
   * @param leftNeurons The Neurons on the left hand side of these Axons
   * @param rightNeurons The Neurons on the right hand side of these Axons
   * @param connectionWeights The connection weights.
   * @param config The config for these Axons.
   */
  public AxonsBase(L leftNeurons, R rightNeurons, MatrixFactory matrixFactory,
      Matrix connectionWeights, C config) {
    this.config = config;
    this.leftNeurons = leftNeurons;
    this.rightNeurons = rightNeurons;
    this.connectionWeightsMask = createConnectionWeightsMask(matrixFactory);
    this.connectionWeights = matrixFactory.createZeros(leftNeurons.getNeuronCountIncludingBias(),
        rightNeurons.getNeuronCountIncludingBias());
    validateConnectionWeightsAndConnectionWeightsMaskDimensions();
    adjustConnectionWeights(connectionWeights, ConnectionWeightsAdjustmentDirection.ADDITION, true);
  }

  /**
   * Construct a new Axons instance.
   * 
   * @param leftNeurons The Neurons on the left hand side of these Axons
   * @param rightNeurons The Neurons on the right hand side of these Axons
   * @param connectionWeights The connection weights Matrix
   * @param connectionWeightsMask The connection weights mask, or null if no mask required.
   */
  protected AxonsBase(L leftNeurons, R rightNeurons, Matrix connectionWeights,
      ConnectionWeightsMask connectionWeightsMask, C config) {
    this.config = config;
    this.leftNeurons = leftNeurons;
    this.rightNeurons = rightNeurons;
    this.connectionWeightsMask = connectionWeightsMask;
    this.connectionWeights = connectionWeights;
    validateConnectionWeightsAndConnectionWeightsMaskDimensions();
  }

  private void validateConnectionWeightsAndConnectionWeightsMaskDimensions() {
    if (connectionWeightsMask != null) {
      if (connectionWeightsMask.getWeightsMask().getRows() != connectionWeights.getRows()) {
        throw new IllegalStateException("Connection weights mask row dimension of "
            + connectionWeightsMask.getWeightsMask().getRows()
            + " does not match the row dimension of the connection weights:"
            + connectionWeights.getRows());
      }
      if (connectionWeightsMask.getWeightsMask().getColumns() != connectionWeights.getColumns()) {
        throw new IllegalStateException("Connection weights mask column dimension of "
            + connectionWeightsMask.getWeightsMask().getColumns()
            + " does not match the column dimension of the connection weights:"
            + connectionWeights.getColumns());
      }
    }
  }

  protected abstract ConnectionWeightsMask createConnectionWeightsMask(MatrixFactory matrixFactory);

  protected abstract Matrix createDefaultInitialConnectionWeights(MatrixFactory matrixFactory);

  @Override
  public L getLeftNeurons() {
    return leftNeurons;
  }

  @Override
  public R getRightNeurons() {
    return rightNeurons;
  }
  
  private Matrix getActivationsWithPossibleBias(Matrix inputMatrix, 
      NeuronsActivationFeatureOrientation orientation, AxonsContext axonsContext, 
      boolean sourceWithBias, boolean targetWithBias, boolean resetTargetBias) {
   
    NeuronsActivationWithPossibleBiasUnit withBiasUnitIfApplicable =
        new NeuronsActivationWithPossibleBiasUnit(inputMatrix,
            sourceWithBias, orientation, 
            resetTargetBias).withBiasUnit(targetWithBias, axonsContext);
    
    inputMatrix = withBiasUnitIfApplicable.getActivationsWithBias();
    
    return inputMatrix;
  }

  @Override
  public AxonsActivation pushLeftToRight(NeuronsActivation leftNeuronsActivation,
      AxonsActivation previousRightToLeftActivation, AxonsContext axonsContext) {
    
    LOGGER.debug("Pushing left to right through Axons");
    if (leftNeuronsActivation
        .getFeatureOrientation() != NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET) {
      throw new IllegalArgumentException("Only neurons actiavation with COLUMNS_SPAN_FEATURE_SET "
          + "orientation supported currently");
    }
    Matrix outputMatrix = null;
    Matrix outputDropoutMask = null;

    Matrix inputDropoutMask = createLeftInputDropoutMask(leftNeuronsActivation, axonsContext);

    Matrix previousInputDropoutMask = previousRightToLeftActivation == null ? null
        : previousRightToLeftActivation.getInputDropoutMask();
    if (previousInputDropoutMask != null) {
      LOGGER.debug("Transposing previous right to left input dropout mask");
      outputDropoutMask = previousInputDropoutMask.transpose();
    }
    
    Matrix inputMatrix = null;

    if (inputDropoutMask != null) {
      double postDropoutScaling = getLeftInputPostDropoutScaling(axonsContext);

      if (postDropoutScaling != 1) {
        LOGGER.debug("Applying input dropout mask");

        LOGGER.debug("Scaling post dropout left to right non-bias input");

        Matrix preScaling = leftNeuronsActivation.getActivations().mul(inputDropoutMask);

        inputMatrix = preScaling.mul(postDropoutScaling);
        
        inputMatrix = getActivationsWithPossibleBias(inputMatrix, 
            leftNeuronsActivation.getFeatureOrientation(), axonsContext, 
            false, leftNeurons.hasBiasUnit(), false);

        outputMatrix = inputMatrix.mmul(connectionWeights);
      } else {
        inputMatrix = leftNeuronsActivation.getActivations().mul(inputDropoutMask);

        inputMatrix = getActivationsWithPossibleBias(inputMatrix, 
            leftNeuronsActivation.getFeatureOrientation(), axonsContext, false, 
            leftNeurons.hasBiasUnit(), false);

        if (inputMatrix.getColumns() != connectionWeights
            .getRows()) {
          throw new IllegalArgumentException("Expected NeuronsActivation should consist of "
              + getLeftNeurons().getNeuronCountIncludingBias()
              + " features including bias but references "
              + inputMatrix.getColumns()
              + " features including bias");
        }
       
        outputMatrix = inputMatrix.mmul(connectionWeights);
      }

    } else {
      
      inputMatrix = getActivationsWithPossibleBias(leftNeuronsActivation.getActivations(), 
          leftNeuronsActivation.getFeatureOrientation(), 
          axonsContext, false, leftNeurons.hasBiasUnit(), false);
     
      if (inputMatrix.getColumns() != connectionWeights
          .getRows()) {
        throw new IllegalArgumentException("Expected NeuronsActivation should consist of "
            + getLeftNeurons().getNeuronCountExcludingBias()
            + " features including bias but references "
            + inputMatrix.getColumns()
            + " features including bias");
      }
      
      outputMatrix = inputMatrix.mmul(connectionWeights);
    }
  
    outputMatrix = 
        getActivationsWithPossibleBias(outputMatrix, 
        leftNeuronsActivation.getFeatureOrientation(), 
        axonsContext, rightNeurons.hasBiasUnit(), false, rightNeurons.hasBiasUnit());

    if (outputDropoutMask != null) {
      LOGGER.debug("Applying left to right output dropout mask");
      outputMatrix = outputMatrix.mul(outputDropoutMask);
    }
       
    NeuronsActivation outputActivation =
        new NeuronsActivation(outputMatrix, 
            leftNeuronsActivation.getFeatureOrientation());

    return new AxonsActivationImpl(this, inputDropoutMask,
        new NeuronsActivationWithPossibleBiasUnit(inputMatrix, leftNeurons.hasBiasUnit(),
            leftNeuronsActivation.getFeatureOrientation(), false),
        outputActivation);
  }

  /**
   * Return the dropout mask for left hand side input.
   * 
   * @param axonsContext The axons context
   * @return The input dropout mask applied at the left hand side of these Axons
   */
  protected Matrix createLeftInputDropoutMask(NeuronsActivation leftNeuronsActivation,
      AxonsContext axonsContext) {

    double leftHandInputDropoutKeepProbability =
        axonsContext.getLeftHandInputDropoutKeepProbability();
    if (leftHandInputDropoutKeepProbability == 1) {
      return null;
    } else {

      LOGGER.debug("Creating left input dropout mask");

      Matrix dropoutMask = axonsContext.getMatrixFactory().createZeros(
          leftNeuronsActivation.getActivations().getRows(),
          leftNeuronsActivation.getActivations().getColumns());
      for (int i = 0; i < dropoutMask.getRows(); i++) {
        for (int j = 0; j < dropoutMask.getColumns(); j++) {
          if (Math.random() < leftHandInputDropoutKeepProbability) {
            dropoutMask.put(i, j, 1);
          }
        }
      }
      return dropoutMask;

    }
  }

  /**
   * Return the scaling required due to left-hand side input dropout.
   * 
   * @param axonsContext The axons context.
   * @return The post dropout input scaling factor.
   */
  protected double getLeftInputPostDropoutScaling(AxonsContext axonsContext) {
    double dropoutKeepProbability = axonsContext.getLeftHandInputDropoutKeepProbability();
    if (dropoutKeepProbability == 0) {
      throw new IllegalArgumentException("Dropout keep probability cannot be set to 0");
    }
    return 1d / dropoutKeepProbability;
  }

  /**
   * Return the scaling required due to right-hand side input dropout. This is not yet supported, so
   * we return 1.
   * 
   * @param axonsContext The axons context.
   * @return The post dropout input scaling factor.
   */
  protected double getRightInputPostDropoutScaling(AxonsContext axonsContext) {
    return 1d;
  }

  /**
   * Return the dropout mask for right hand side input. This is not yet supported, so we return
   * null.
   * 
   * @param axonsContext The axons context
   * @return The input dropout mask applied at the right hand side of these Axons
   */
  protected Matrix createRightInputDropoutMask(NeuronsActivation rightNeuronsActivation,
      AxonsContext axonsContext) {
    return null;
  }

  @Override
  public AxonsActivation pushRightToLeft(NeuronsActivation rightNeuronsActivation,
      AxonsActivation previousLeftToRightActivation, AxonsContext axonsContext) {
   
    LOGGER.debug("Pushing right to left through Axons:");
    if (rightNeuronsActivation
        .getFeatureOrientation() != NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET) {
      throw new IllegalArgumentException("Only neurons actiavation with ROWS_SPAN_FEATURE_SET "
          + "orientation supported currently");
    }

    Matrix outputMatrix = null;
    Matrix outputDropoutMask = null;

    Matrix inputDropoutMask = createRightInputDropoutMask(rightNeuronsActivation, axonsContext);

    Matrix previousInputDropoutMask = previousLeftToRightActivation == null ? null
        : previousLeftToRightActivation.getInputDropoutMask();
    if (previousInputDropoutMask != null) {
      LOGGER.debug("Transposing previous input dropout mask");
      outputDropoutMask = previousInputDropoutMask.transpose();
    }
    
    Matrix inputMatrix = null;
    if (inputDropoutMask != null) {
      double postDropoutScaling = getRightInputPostDropoutScaling(axonsContext);
      if (postDropoutScaling != 1) {
        LOGGER.debug("Scaling post dropout right to left non-bias input");

        Matrix preScaling = rightNeuronsActivation
            .getActivations().mul(inputDropoutMask);
        
        inputMatrix = preScaling.mul(postDropoutScaling);
        
        inputMatrix = getActivationsWithPossibleBias(inputMatrix, 
            rightNeuronsActivation.getFeatureOrientation(), 
            axonsContext, false, rightNeurons.hasBiasUnit(), false);

        if (inputMatrix.getRows() != connectionWeights
            .getColumns()) {
          throw new IllegalArgumentException("Expected NeuronsActivation should consist of "
              + getRightNeurons().getNeuronCountIncludingBias()
              + " features including bias but references "
              + inputMatrix.getRows()
              + " features including bias");
        }
        
        outputMatrix = connectionWeights.mmul(inputMatrix);
      } else {
        
        
        inputMatrix = rightNeuronsActivation.getActivations()
            .mul(inputDropoutMask);
        
        inputMatrix = getActivationsWithPossibleBias(inputMatrix, 
            rightNeuronsActivation.getFeatureOrientation(), 
            axonsContext, false, rightNeurons.hasBiasUnit(), false);
        
        if (inputMatrix.getRows() != connectionWeights
            .getColumns()) {
          throw new IllegalArgumentException("Expected NeuronsActivation should consist of "
              + getRightNeurons().getNeuronCountIncludingBias()
              + " features including bias but references "
              + inputMatrix.getRows()
              + " features including bias");
        }
                
        outputMatrix = connectionWeights.mmul(inputMatrix);
      }
    } else {
    
      inputMatrix = getActivationsWithPossibleBias(
          rightNeuronsActivation.getActivations(), 
          rightNeuronsActivation.getFeatureOrientation(), 
          axonsContext, false, rightNeurons.hasBiasUnit(), false);
     
      if (inputMatrix.getRows() != connectionWeights
          .getColumns()) {
        throw new IllegalArgumentException("Expected NeuronsActivation should consist of "
            + getRightNeurons().getNeuronCountIncludingBias()
            + " features including bias but references "
            + inputMatrix.getRows()
            + " features including bias");
      }
           
      outputMatrix = connectionWeights.mmul(inputMatrix);
    }
   
    outputMatrix = 
        getActivationsWithPossibleBias(outputMatrix, 
        rightNeuronsActivation.getFeatureOrientation(), 
        axonsContext, leftNeurons.hasBiasUnit(), false, leftNeurons.hasBiasUnit());
    
    if (outputDropoutMask != null) {
      LOGGER.debug("Applying right to left output dropout mask");
      outputMatrix = outputMatrix.mul(outputDropoutMask);
    }
        
    NeuronsActivation outputActivation =
        new NeuronsActivation(outputMatrix,
            rightNeuronsActivation.getFeatureOrientation());

    return new AxonsActivationImpl(this, inputDropoutMask,
        new NeuronsActivationWithPossibleBiasUnit(inputMatrix, 
            rightNeurons.hasBiasUnit(),
            rightNeuronsActivation.getFeatureOrientation(), false),
        outputActivation);
  }

  protected void adjustConnectionWeights(Matrix adjustment,
      ConnectionWeightsAdjustmentDirection adjustmentDirection, boolean initialisation) {

    if (adjustment.getRows() != connectionWeights.getRows()
        || adjustment.getColumns() != connectionWeights.getColumns()) {
      throw new IllegalArgumentException(
          "Connection weights adjustment matrix is of dimensions: " + adjustment.getRows() + ","
              + adjustment.getColumns() + " but connection weights matrix is of dimensions:"
              + connectionWeights.getRows() + "," + connectionWeights.getColumns());
    }

    if (connectionWeightsMask != null) {
      LOGGER.debug("Applying connection weights mask to adjustment request");
      adjustment.muli(connectionWeightsMask.getWeightsMask());
    }
    applyAdditionalConnectionWeightAdjustmentConstraints(adjustment);
    if (adjustmentDirection == ConnectionWeightsAdjustmentDirection.ADDITION) {
      LOGGER.debug("Adding adjustment to connection weights");
      connectionWeights.addi(adjustment);
    } else {
      LOGGER.debug("Subtracting adjustment from connection weights");
      connectionWeights.subi(adjustment);
    }
  }
  
  
  @Override
  public Matrix getDetachedConnectionWeights() {
    LOGGER.debug("Duplicating connetion weights");
    return connectionWeights.dup();
  }
  

  protected void applyAdditionalConnectionWeightAdjustmentConstraints(Matrix adjustment) {
    // No-op by default
  }
}
