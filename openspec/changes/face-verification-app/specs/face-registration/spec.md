## Purpose

Let the user register a named person's face from one or more photos and manage that registration (replace or delete it).

## ADDED Requirements

### Requirement: Register a person from photos

The system SHALL allow the user to register a named person by selecting one or more face photos. The system SHALL extract a face feature vector from each photo and store a single reference derived from all of them, associated with the provided name.

#### Scenario: Successful registration

- **WHEN** the user selects one or more photos that each contain a detectable face and provides a name
- **THEN** the system stores the registration and displays the name and the number of photos used

#### Scenario: A selected photo has no detectable face

- **WHEN** the user selects a photo that contains no detectable face
- **THEN** the system reports that the photo has no face and does not include it in the registration

#### Scenario: No photo has a detectable face

- **WHEN** none of the selected photos contain a detectable face
- **THEN** the system reports that registration failed and stores nothing

### Requirement: Replace an existing registration

The system SHALL allow replacing an existing registration with a new one, and SHALL require user confirmation before overwriting.

#### Scenario: Replace with confirmation

- **WHEN** a registration already exists and the user initiates a new registration
- **THEN** the system prompts for confirmation before overwriting the existing registration

#### Scenario: Cancel a replacement

- **WHEN** the user declines the overwrite confirmation
- **THEN** the existing registration is left unchanged

### Requirement: Delete a registration

The system SHALL allow deleting the current registration, and SHALL require user confirmation before clearing it.

#### Scenario: Delete registration

- **WHEN** the user deletes the registration and confirms
- **THEN** the system clears the stored registration and shows an unregistered state

### Requirement: Persist registration across restarts

The system SHALL persist the current registration to local storage so that it survives an application restart.

#### Scenario: Registration survives restart

- **WHEN** a registration exists and the application is restarted
- **THEN** the stored registration (name and reference vector) is available on the next launch
