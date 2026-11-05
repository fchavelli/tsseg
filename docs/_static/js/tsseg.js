/* tsseg documentation: small helpers of the landing page */
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("[data-ts-copy]").forEach((button) => {
    const label = button.querySelector("span");
    button.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(button.dataset.tsCopy);
      } catch {
        return;
      }
      button.classList.add("is-done");
      if (label) label.textContent = "Copied";
      setTimeout(() => {
        button.classList.remove("is-done");
        if (label) label.textContent = "Copy";
      }, 1600);
    });
  });
});
