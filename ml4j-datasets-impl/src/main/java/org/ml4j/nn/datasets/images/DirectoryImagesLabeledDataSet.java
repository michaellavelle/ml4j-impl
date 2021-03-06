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
package org.ml4j.nn.datasets.images;

import java.nio.file.Path;
import java.util.function.Supplier;

import org.ml4j.images.Image;
import org.ml4j.nn.datasets.DataLabeler;
import org.ml4j.nn.datasets.DataSetImpl;
import org.ml4j.nn.datasets.LabeledDataSetImpl;

public class DirectoryImagesLabeledDataSet<L> extends LabeledDataSetImpl<Supplier<Image>, L>
		implements LabeledImagesDataSet<L> {

	public DirectoryImagesLabeledDataSet(DirectoryImagesWithPathsDataSet labeledDataSet,
			DataLabeler<Path, L> dataLabeler) {
		super(new DataSetImpl<>(() -> labeledDataSet.stream()), dataLabeler);
	}

	@Override
	public ImagesDataSet getDataSet() {
		return new ImagesDataSetImpl(() -> super.getDataSet().stream());
	}
}
