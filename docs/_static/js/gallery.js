document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".tsseg-gallery").forEach((gallery) => {
    const buttons = gallery.querySelectorAll("button[data-filter]");
    const cards = gallery.querySelectorAll(".tsseg-card");
    const status = gallery.querySelector(".tsseg-filter-status");
    buttons.forEach((button) => {
      button.addEventListener("click", () => {
        const filter = button.dataset.filter;
        buttons.forEach((item) => item.setAttribute("aria-pressed", String(item === button)));
        let visible = 0;
        cards.forEach((card) => {
          const task = card.dataset.task;
          card.hidden = !(filter === "all" || task === filter);
          if (!card.hidden) visible += 1;
        });
        status.textContent = `${visible} ${visible === 1 ? "example" : "examples"}`;
      });
    });
  });
});
