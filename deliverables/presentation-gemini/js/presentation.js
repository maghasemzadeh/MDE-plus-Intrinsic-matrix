/**
 * Thesis Defense Presentation - Interactive Controller
 * Features:
 * - Arrow & Space keyboard navigation
 * - Presenter Notes Drawer (Key: 'N')
 * - Full Overview Grid (Key: 'O' or 'G')
 * - Fullscreen mode (Key: 'F')
 * - Progress bar & slide counter (X / 43)
 * - Auto video play/pause on slide transition
 */

document.addEventListener('DOMContentLoaded', () => {
  const slides = Array.from(document.querySelectorAll('.slide'));
  const totalSlides = slides.length;
  let currentIndex = 0;

  // UI Elements
  const counterElement = document.getElementById('nav-counter');
  const progressBar = document.getElementById('progress-bar');
  const notesDrawer = document.getElementById('notes-drawer');
  const notesBody = document.getElementById('notes-body');
  const overviewModal = document.getElementById('overview-modal');
  const overviewGrid = document.getElementById('overview-grid');
  
  const btnPrev = document.getElementById('btn-prev');
  const btnNext = document.getElementById('btn-next');
  const btnNotes = document.getElementById('btn-notes');
  const btnOverview = document.getElementById('btn-overview');
  const btnFullscreen = document.getElementById('btn-fullscreen');
  const btnCloseNotes = document.getElementById('btn-close-notes');
  const btnCloseOverview = document.getElementById('btn-close-overview');

  // Build Overview Grid Cards
  function buildOverviewGrid() {
    overviewGrid.innerHTML = '';
    slides.forEach((slide, index) => {
      const card = document.createElement('div');
      card.className = `overview-card ${index === currentIndex ? 'current' : ''}`;
      
      const titleEl = slide.querySelector('.slide-header h2') || slide.querySelector('.thesis-title');
      const titleText = titleEl ? titleEl.textContent.trim() : `اسلاید ${index + 1}`;
      
      card.innerHTML = `
        <div class="num">${index + 1} / ${totalSlides}</div>
        <div class="title">${titleText}</div>
      `;
      
      card.addEventListener('click', () => {
        goToSlide(index);
        closeOverview();
      });
      
      overviewGrid.appendChild(card);
    });
  }

  // Update Slide View
  function goToSlide(index) {
    if (index < 0 || index >= totalSlides) return;

    // Handle video on leaving previous slide
    const currentSlide = slides[currentIndex];
    const prevVideo = currentSlide.querySelector('video');
    if (prevVideo) {
      prevVideo.pause();
    }

    // Update classes
    slides[currentIndex].classList.remove('active');
    currentIndex = index;
    const newSlide = slides[currentIndex];
    newSlide.classList.add('active');

    // Handle video on entering new slide
    const newVideo = newSlide.querySelector('video');
    if (newVideo) {
      newVideo.currentTime = 0;
      newVideo.play().catch(() => {});
    }

    // Update Counter & Progress Bar
    const slideNumberStr = `${currentIndex + 1} / ${totalSlides}`;
    if (counterElement) counterElement.textContent = slideNumberStr;
    if (progressBar) {
      const percent = ((currentIndex + 1) / totalSlides) * 100;
      progressBar.style.width = `${percent}%`;
    }

    // Update Speaker Notes
    updateSpeakerNotes();

    // Update Overview highlight
    const overviewCards = overviewGrid.querySelectorAll('.overview-card');
    overviewCards.forEach((c, idx) => {
      c.classList.toggle('current', idx === currentIndex);
    });

    // Save hash or state
    window.location.hash = `slide-${currentIndex + 1}`;
  }

  function nextSlide() {
    if (currentIndex < totalSlides - 1) {
      goToSlide(currentIndex + 1);
    }
  }

  function prevSlide() {
    if (currentIndex > 0) {
      goToSlide(currentIndex - 1);
    }
  }

  // Speaker Notes Handling
  function updateSpeakerNotes() {
    const activeSlide = slides[currentIndex];
    const notesContent = activeSlide.querySelector('.speaker-notes');
    if (notesContent && notesBody) {
      notesBody.innerHTML = notesContent.innerHTML;
    } else if (notesBody) {
      notesBody.innerHTML = '<p><em>یادداشتی برای این اسلاید ثبت نشده است.</em></p>';
    }
  }

  function toggleNotes() {
    notesDrawer.classList.toggle('open');
  }

  function closeNotes() {
    notesDrawer.classList.remove('open');
  }

  // Overview Handling
  function toggleOverview() {
    overviewModal.classList.toggle('open');
  }

  function closeOverview() {
    overviewModal.classList.remove('open');
  }

  // Fullscreen Handling
  function toggleFullscreen() {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().catch(() => {});
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen();
      }
    }
  }

  // Event Listeners for Controls
  if (btnNext) btnNext.addEventListener('click', nextSlide);
  if (btnPrev) btnPrev.addEventListener('click', prevSlide);
  if (btnNotes) btnNotes.addEventListener('click', toggleNotes);
  if (btnOverview) btnOverview.addEventListener('click', toggleOverview);
  if (btnFullscreen) btnFullscreen.addEventListener('click', toggleFullscreen);
  if (btnCloseNotes) btnCloseNotes.addEventListener('click', closeNotes);
  if (btnCloseOverview) btnCloseOverview.addEventListener('click', closeOverview);

  // Keyboard Navigation
  document.addEventListener('keydown', (e) => {
    // If overview is open, Escape closes it
    if (overviewModal.classList.contains('open')) {
      if (e.key === 'Escape' || e.key === 'o' || e.key === 'O' || e.key === 'g' || e.key === 'G') {
        closeOverview();
        e.preventDefault();
      }
      return;
    }

    switch (e.key) {
      case 'ArrowLeft': // Next in RTL / standard forward
      case 'PageDown':
      case ' ': // Spacebar
      case 'j':
      case 'J':
        nextSlide();
        e.preventDefault();
        break;

      case 'ArrowRight': // Prev in RTL / standard backward
      case 'PageUp':
      case 'Backspace':
      case 'k':
      case 'K':
        prevSlide();
        e.preventDefault();
        break;

      case 'Home':
        goToSlide(0);
        e.preventDefault();
        break;

      case 'End':
        goToSlide(totalSlides - 1);
        e.preventDefault();
        break;

      case 'n':
      case 'N':
        toggleNotes();
        e.preventDefault();
        break;

      case 'o':
      case 'O':
      case 'g':
      case 'G':
        toggleOverview();
        e.preventDefault();
        break;

      case 'f':
      case 'F':
        toggleFullscreen();
        e.preventDefault();
        break;

      case 'Escape':
        closeNotes();
        closeOverview();
        break;
    }
  });

  // Check initial hash
  const hash = window.location.hash;
  if (hash && hash.startsWith('#slide-')) {
    const slideNum = parseInt(hash.replace('#slide-', ''), 10);
    if (!isNaN(slideNum) && slideNum >= 1 && slideNum <= totalSlides) {
      currentIndex = slideNum - 1;
    }
  }

  // Initialize
  buildOverviewGrid();
  goToSlide(currentIndex);
});
