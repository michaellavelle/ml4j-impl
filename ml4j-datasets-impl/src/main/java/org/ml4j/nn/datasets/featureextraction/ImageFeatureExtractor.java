/*
 * Copyright 2019 the original author or authors.
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
package org.ml4j.nn.datasets.featureextraction;

import org.ml4j.images.Image;
import org.ml4j.nn.datasets.FeatureExtractor;
import org.ml4j.nn.datasets.exceptions.FeatureExtractionException;

public class ImageFeatureExtractor implements FeatureExtractor<Image> {

	private int featureCount;

	public ImageFeatureExtractor(int featureCount) {
		this.featureCount = featureCount;
	}

	@Override
	public float[] getFeatures(Image data) throws FeatureExtractionException {
		float[] features = data.getData();
		if (features == null) {
			throw new FeatureExtractionException("Image data was null");
		}
		return features;
	}

	@Override
	public int getFeatureCount() {
		return featureCount;
	}

}
