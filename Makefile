# TODO
run:
	uvicorn sign_game.api.fast:app --host 0.0.0.0


# delete the existing augmented set
clean_output:
	echo $(REPO)$(OUTPUT_PATH)
	@rm -r $(REPO)$(OUTPUT_PATH)

# make a test image set to run our augmentation on
create_test_images_set:
	@mkdir $(TEST_IMAGES)
	@find $(SOURCE_IMAGES) -type f | head -50 | xargs cp -t $(TEST_IMAGES)

# check the number of images in the unaugmented test dataset
check_test_images_set:
	@ls -1 $(TEST_IMAGES)	| wc -l

# run the script to generate images in the relevant directory
create_dataset:
	if [ ! -d $(OUTPUT_PATH) ]; then mkdir $(OUTPUT_PATH); echo 'made output'; fi
	@python image-augmentation/image-augmentation.py
	@ls $(OUTPUT_PATH)

# rename_data ----  rename -n 's/.{27}(.)/$1/' *
