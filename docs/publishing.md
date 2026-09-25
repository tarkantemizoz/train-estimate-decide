# Publish and use the GitHub template

## Organizer

Follow these steps when publishing a new copy of the tutorial as a GitHub template.

1. Create an empty repository under your account or the DynaPlex organization, with the intended name and visibility.
2. Add the contents of this folder to its repository root, including `.github/`, `.gitignore`, and `.gitattributes`.
   Keep `README.md` and `train.py` at the repository root rather than inside an extra `ted-tutorial/` folder.
3. Add the project's chosen license and commit/push the files.
4. Check the GitHub Actions training checks, then open **Settings > Template repository** and select the checkbox.
5. Share the repository URL with attendees and ask them to use **Use this template**.

Only the prepared template folder belongs in the published repository.
Local virtual environments, generated `runs/`, the legacy C++ repository, and development studies are not part of the distribution.
GitHub requires repository administrator permissions to enable the template setting; see its [template creation instructions](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-template-repository).

## Participant

1. Open the organizer's repository and select **Use this template > Create a new repository**.
2. Choose the owner, repository name, and visibility, then create the repository.
3. Clone your newly created repository or download and extract its ZIP.
4. Open a terminal in that directory and follow the [README](../README.md): install, smoke check, train, and evaluate.
5. Save your experiment configurations in your own repository and keep generated outputs under `runs/`.

Creating a repository from a template gives you an independent repository with its own history.
It does not automatically receive later changes from the organizer's template.
See GitHub's [participant instructions](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-repository-from-a-template).
