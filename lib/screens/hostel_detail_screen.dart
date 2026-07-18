import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../api.dart';
import '../theme.dart';
import '../widgets/responsive_shell.dart';
import 'booking_screen.dart';

class HostelDetailScreen extends StatefulWidget {
  final int id;
  const HostelDetailScreen({super.key, required this.id});

  @override
  State<HostelDetailScreen> createState() => _HostelDetailScreenState();
}

class _HostelDetailScreenState extends State<HostelDetailScreen> {
  final api = Api();
  Map<String, dynamic>? hostel;
  bool _isFav = false;
  bool _savingFavourite = false;

  @override
  void initState() {
    super.initState();
    _loadHostel();
  }

  Future<void> _loadHostel() async {
    try {
      final data = await api.getHostelDetail(widget.id);
      var isFavourite = false;
      try {
        final favourites = await api.getFavourites();
        isFavourite = favourites.any(
          (favorite) =>
              favorite['hostel_id']?.toString() == widget.id.toString(),
        );
      } catch (_) {
        // A hostel remains browseable when the visitor is not signed in.
      }
      if (mounted)
        setState(() {
          hostel = data;
          _isFav = isFavourite;
        });
    } catch (error) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('$error'),
            backgroundColor: AppTheme.teal,
            behavior: SnackBarBehavior.floating,
          ),
        );
      }
    }
  }

  Future<void> _toggleFavourite() async {
    if (_savingFavourite) return;
    setState(() => _savingFavourite = true);
    try {
      if (_isFav) {
        await api.removeFavourite(widget.id);
      } else {
        await api.addFavourite(widget.id);
      }
      if (mounted) setState(() => _isFav = !_isFav);
    } catch (error) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(error.toString(), style: GoogleFonts.dmSans()),
            backgroundColor: Colors.red.shade600,
            behavior: SnackBarBehavior.floating,
          ),
        );
      }
    } finally {
      if (mounted) setState(() => _savingFavourite = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return ResponsiveShell(
      child: Scaffold(
        backgroundColor: AppTheme.bg,
        body: hostel == null
            ? const Center(
                child: CircularProgressIndicator(color: AppTheme.teal),
              )
            : _buildContent(context),
      ),
    );
  }

  Widget _buildContent(BuildContext context) {
    final name = hostel!['name'] ?? 'Hostel';
    final city = hostel!['city'] ?? '';
    final address = hostel!['address'] ?? '';
    final description = hostel!['description'] ?? '';
    final avail = hostel!['available_capacity'] ?? 0;
    final total = hostel!['total_capacity'] ?? 0;
    final reviews = (hostel!['reviews'] ?? []) as List;

    return Column(
      children: [
        Expanded(
          child: CustomScrollView(
            slivers: [
              // Hero
              SliverToBoxAdapter(child: _buildHero(context, name, city)),

              // Content
              SliverPadding(
                padding: const EdgeInsets.fromLTRB(20, 20, 20, 0),
                sliver: SliverList(
                  delegate: SliverChildListDelegate([
                    // Address
                    if (address.isNotEmpty) ...[
                      Row(
                        children: [
                          Icon(
                            Icons.location_on_rounded,
                            size: 14,
                            color: AppTheme.textMuted,
                          ),
                          const SizedBox(width: 4),
                          Expanded(
                            child: Text(
                              address,
                              style: GoogleFonts.dmSans(
                                fontSize: 12,
                                color: AppTheme.textMuted,
                              ),
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 20),
                    ],

                    // Stat cards
                    _buildStatGrid(avail, total),
                    const SizedBox(height: 24),

                    // Description
                    if (description.isNotEmpty) ...[
                      _sectionLabel('About'),
                      const SizedBox(height: 8),
                      Text(
                        description,
                        style: GoogleFonts.dmSans(
                          fontSize: 14,
                          color: AppTheme.textSecondary,
                          height: 1.55,
                        ),
                      ),
                      const SizedBox(height: 24),
                    ],

                    // Amenities (static showcase)
                    _sectionLabel('Amenities'),
                    const SizedBox(height: 10),
                    _buildAmenities(),
                    const SizedBox(height: 24),

                    // Reviews
                    _sectionLabel('Reviews (${reviews.length})'),
                    const SizedBox(height: 12),
                    ...reviews.map((r) => _ReviewCard(review: r)),

                    if (reviews.isEmpty)
                      Container(
                        padding: const EdgeInsets.all(20),
                        decoration: BoxDecoration(
                          color: AppTheme.card,
                          borderRadius: BorderRadius.circular(14),
                          border: Border.all(
                            color: AppTheme.border,
                            width: 0.5,
                          ),
                        ),
                        child: Center(
                          child: Text(
                            'No reviews yet.',
                            style: GoogleFonts.dmSans(
                              color: AppTheme.textMuted,
                              fontSize: 13,
                            ),
                          ),
                        ),
                      ),

                    const SizedBox(height: 20),
                  ]),
                ),
              ),
            ],
          ),
        ),

        // Sticky bottom bar
        _buildBottomBar(),
      ],
    );
  }

  Widget _buildHero(BuildContext context, String name, String city) {
    return Stack(
      children: [
        Container(
          height: 230,
          decoration: const BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: [Color(0xFF1D9E75), AppTheme.teal, AppTheme.tealDeep],
            ),
          ),
          child: SafeArea(
            child: Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Container(
                    width: 70,
                    height: 70,
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.15),
                      shape: BoxShape.circle,
                    ),
                    child: const Icon(
                      Icons.business_rounded,
                      size: 34,
                      color: Colors.white,
                    ),
                  ),
                  const SizedBox(height: 12),
                  Text(
                    name,
                    style: GoogleFonts.playfairDisplay(
                      fontSize: 22,
                      fontWeight: FontWeight.w700,
                      color: Colors.white,
                    ),
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 4),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      const Icon(
                        Icons.location_on_rounded,
                        size: 14,
                        color: Colors.white70,
                      ),
                      const SizedBox(width: 3),
                      Text(
                        city,
                        style: GoogleFonts.dmSans(
                          fontSize: 13,
                          color: Colors.white.withOpacity(0.75),
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),
        ),

        // Back button
        Positioned(
          top: MediaQuery.of(context).padding.top + 8,
          left: 12,
          child: _heroButton(
            child: const Icon(
              Icons.arrow_back_rounded,
              size: 20,
              color: Colors.white,
            ),
            onTap: () => Navigator.pop(context),
          ),
        ),

        // Favourite button
        Positioned(
          top: MediaQuery.of(context).padding.top + 8,
          right: 12,
          child: _heroButton(
            child: Icon(
              _isFav ? Icons.favorite_rounded : Icons.favorite_border_rounded,
              size: 20,
              color: _isFav ? Colors.redAccent : Colors.white,
            ),
            onTap: () {
              if (!_savingFavourite) _toggleFavourite();
            },
          ),
        ),
      ],
    );
  }

  Widget _heroButton({required Widget child, required VoidCallback onTap}) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: 38,
        height: 38,
        decoration: BoxDecoration(
          color: Colors.black.withOpacity(0.22),
          shape: BoxShape.circle,
          border: Border.all(color: Colors.white.withOpacity(0.2)),
        ),
        child: Center(child: child),
      ),
    );
  }

  Widget _buildStatGrid(int avail, int total) {
    final occupancy = total > 0 ? ((total - avail) / total * 100).round() : 0;
    return Row(
      children: [
        _StatCard(value: '$avail', label: 'Available', icon: Icons.bed_rounded),
        const SizedBox(width: 10),
        _StatCard(
          value: '$total',
          label: 'Capacity',
          icon: Icons.people_rounded,
        ),
        const SizedBox(width: 10),
        _StatCard(
          value: '$occupancy%',
          label: 'Occupied',
          icon: Icons.pie_chart_rounded,
          highlight: occupancy > 80,
        ),
      ],
    );
  }

  Widget _sectionLabel(String text) {
    return Text(
      text,
      style: GoogleFonts.dmSans(
        fontSize: 15,
        fontWeight: FontWeight.w700,
        color: AppTheme.textPrimary,
      ),
    );
  }

  Widget _buildAmenities() {
    const amenities = [
      {'icon': Icons.wifi_rounded, 'label': 'WiFi'},
      {'icon': Icons.water_drop_rounded, 'label': 'Water'},
      {'icon': Icons.bolt_rounded, 'label': 'Power'},
      {'icon': Icons.security_rounded, 'label': 'Security'},
      {'icon': Icons.local_parking_rounded, 'label': 'Parking'},
      {'icon': Icons.restaurant_rounded, 'label': 'Meals'},
    ];

    return Wrap(
      spacing: 8,
      runSpacing: 8,
      children: amenities.map((a) {
        return Container(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 7),
          decoration: BoxDecoration(
            color: AppTheme.teal.withOpacity(0.07),
            borderRadius: BorderRadius.circular(10),
            border: Border.all(
              color: AppTheme.teal.withOpacity(0.18),
              width: 0.5,
            ),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(a['icon'] as IconData, size: 15, color: AppTheme.teal),
              const SizedBox(width: 5),
              Text(
                a['label'] as String,
                style: GoogleFonts.dmSans(
                  fontSize: 12,
                  fontWeight: FontWeight.w500,
                  color: AppTheme.teal,
                ),
              ),
            ],
          ),
        );
      }).toList(),
    );
  }

  void _showReviewDialog() {
    var rating = 5.0;
    final textController = TextEditingController();
    showDialog(
      context: context,
      builder: (dialogContext) => StatefulBuilder(
        builder: (context, setDialogState) => AlertDialog(
          title: const Text('Write a Review'),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text('${rating.toStringAsFixed(1)} / 5'),
              Slider(
                value: rating,
                min: 1,
                max: 5,
                divisions: 8,
                label: rating.toStringAsFixed(1),
                onChanged: (value) => setDialogState(() => rating = value),
              ),
              TextField(
                controller: textController,
                maxLines: 3,
                decoration: const InputDecoration(
                  labelText: 'Your experience (optional)',
                  border: OutlineInputBorder(),
                ),
              ),
            ],
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(dialogContext),
              child: const Text('Cancel'),
            ),
            ElevatedButton(
              onPressed: () async {
                try {
                  await api.submitReview(
                    hostelId: widget.id,
                    overallRating: rating,
                    textReview: textController.text.trim().isEmpty
                        ? null
                        : textController.text.trim(),
                  );
                  if (!mounted) return;
                  Navigator.pop(dialogContext);
                  await _loadHostel();
                } catch (error) {
                  if (!mounted) return;
                  ScaffoldMessenger.of(context).showSnackBar(
                    SnackBar(
                      content: Text(
                        error.toString(),
                        style: GoogleFonts.dmSans(),
                      ),
                      backgroundColor: Colors.red.shade600,
                    ),
                  );
                }
              },
              child: const Text('Submit'),
            ),
          ],
        ),
      ),
    ).whenComplete(textController.dispose);
  }

  void _showComplaintDialog() {
    final descriptionController = TextEditingController();
    const categories = [
      'Other',
      'Maintenance',
      'Cleanliness',
      'Food Quality',
      'Safety/Security',
      'Noise/Disturbance',
    ];
    var category = categories.first;
    var severity = 'medium';
    showDialog(
      context: context,
      builder: (dialogContext) => StatefulBuilder(
        builder: (context, setDialogState) => AlertDialog(
          title: const Text('Report an Issue'),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              DropdownButtonFormField<String>(
                value: category,
                decoration: const InputDecoration(labelText: 'Category'),
                items: categories
                    .map(
                      (item) =>
                          DropdownMenuItem(value: item, child: Text(item)),
                    )
                    .toList(),
                onChanged: (value) => setDialogState(() => category = value!),
              ),
              DropdownButtonFormField<String>(
                value: severity,
                decoration: const InputDecoration(labelText: 'Severity'),
                items: const [
                  DropdownMenuItem(value: 'low', child: Text('Low')),
                  DropdownMenuItem(value: 'medium', child: Text('Medium')),
                  DropdownMenuItem(value: 'high', child: Text('High')),
                ],
                onChanged: (value) => setDialogState(() => severity = value!),
              ),
              const SizedBox(height: 12),
              TextField(
                controller: descriptionController,
                maxLines: 4,
                decoration: const InputDecoration(
                  labelText: 'Describe the issue',
                  border: OutlineInputBorder(),
                ),
              ),
            ],
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(dialogContext),
              child: const Text('Cancel'),
            ),
            ElevatedButton(
              onPressed: () async {
                if (descriptionController.text.trim().isEmpty) return;
                try {
                  await api.fileComplaint(
                    hostelId: widget.id,
                    category: category,
                    severity: severity,
                    description: descriptionController.text.trim(),
                  );
                  if (!mounted) return;
                  Navigator.pop(dialogContext);
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(
                      content: Text('Issue reported successfully.'),
                    ),
                  );
                } catch (error) {
                  if (!mounted) return;
                  ScaffoldMessenger.of(context).showSnackBar(
                    SnackBar(
                      content: Text(
                        error.toString(),
                        style: GoogleFonts.dmSans(),
                      ),
                      backgroundColor: Colors.red.shade600,
                    ),
                  );
                }
              },
              child: const Text('Submit'),
            ),
          ],
        ),
      ),
    ).whenComplete(descriptionController.dispose);
  }

  Widget _buildBottomBar() {
    return Container(
      padding: EdgeInsets.fromLTRB(
        20,
        14,
        20,
        14 + MediaQuery.of(context).padding.bottom,
      ),
      decoration: BoxDecoration(
        color: AppTheme.card,
        border: Border(top: BorderSide(color: AppTheme.border, width: 0.5)),
      ),
      child: Row(
        children: [
          Expanded(
            child: SizedBox(
              height: 50,
              child: ElevatedButton(
                onPressed: (hostel!['available_capacity'] as num? ?? 0) > 0
                    ? () => Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (_) => BookingScreen(
                            hostelId: widget.id,
                            hostelName: hostel!['name'] ?? 'Hostel',
                            city: hostel!['city'] ?? '',
                            address: hostel!['address'] ?? '',
                          ),
                        ),
                      )
                    : null,
                style: ElevatedButton.styleFrom(
                  backgroundColor: AppTheme.teal,
                  foregroundColor: Colors.white,
                  elevation: 0,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(14),
                  ),
                ),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const Icon(Icons.hotel_rounded, size: 18),
                    const SizedBox(width: 8),
                    Text(
                      (hostel!['available_capacity'] as num? ?? 0) > 0
                          ? 'Request Capacity'
                          : 'No Capacity Available',
                      style: GoogleFonts.dmSans(
                        fontSize: 15,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
          const SizedBox(width: 12),
          Container(
            width: 50,
            height: 50,
            decoration: BoxDecoration(
              color: AppTheme.bgSecondary,
              borderRadius: BorderRadius.circular(14),
              border: Border.all(color: AppTheme.border, width: 0.5),
            ),
            child: PopupMenuButton<String>(
              tooltip: 'More actions',
              icon: const Icon(
                Icons.more_horiz_rounded,
                size: 20,
                color: AppTheme.textSecondary,
              ),
              onSelected: (action) {
                if (action == 'review') _showReviewDialog();
                if (action == 'complaint') _showComplaintDialog();
              },
              itemBuilder: (_) => const [
                PopupMenuItem(
                  value: 'review',
                  child: ListTile(
                    leading: Icon(Icons.rate_review_outlined),
                    title: Text('Write a review'),
                  ),
                ),
                PopupMenuItem(
                  value: 'complaint',
                  child: ListTile(
                    leading: Icon(Icons.report_outlined),
                    title: Text('Report an issue'),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _StatCard extends StatelessWidget {
  final String value;
  final String label;
  final IconData icon;
  final bool highlight;

  const _StatCard({
    required this.value,
    required this.label,
    required this.icon,
    this.highlight = false,
  });

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 10),
        decoration: BoxDecoration(
          color: AppTheme.card,
          borderRadius: BorderRadius.circular(14),
          border: Border.all(
            color: highlight ? AppTheme.teal.withOpacity(0.3) : AppTheme.border,
            width: 0.5,
          ),
        ),
        child: Column(
          children: [
            Icon(icon, size: 20, color: AppTheme.teal),
            const SizedBox(height: 6),
            Text(
              value,
              style: GoogleFonts.dmSans(
                fontSize: 18,
                fontWeight: FontWeight.w700,
                color: AppTheme.teal,
              ),
            ),
            Text(
              label,
              style: GoogleFonts.dmSans(
                fontSize: 11,
                color: AppTheme.textMuted,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _ReviewCard extends StatelessWidget {
  final dynamic review;
  const _ReviewCard({required this.review});

  @override
  Widget build(BuildContext context) {
    final rating = (review['overall_rating'] ?? 0) as num;
    final text = review['text_review'] ?? '';

    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Container(
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: AppTheme.card,
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: AppTheme.border, width: 0.5),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Stars
            Row(
              children: [
                ...List.generate(5, (i) {
                  return Icon(
                    i < rating.round()
                        ? Icons.star_rounded
                        : Icons.star_outline_rounded,
                    size: 16,
                    color: const Color(0xFFEF9F27),
                  );
                }),
                const SizedBox(width: 8),
                Text(
                  '$rating / 5',
                  style: GoogleFonts.dmSans(
                    fontSize: 12,
                    fontWeight: FontWeight.w600,
                    color: const Color(0xFFBA7517),
                  ),
                ),
              ],
            ),
            if (text.isNotEmpty) ...[
              const SizedBox(height: 8),
              Text(
                text,
                style: GoogleFonts.dmSans(
                  fontSize: 13,
                  color: AppTheme.textSecondary,
                  height: 1.5,
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}
