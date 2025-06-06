from sqlalchemy import create_engine, Column, Integer, String, Float, Date, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class Bank(Base):
    """Model for storing bank information."""
    __tablename__ = 'banks'

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, unique=True)
    package_name = Column(String(100), nullable=False)
    short_name = Column(String(10), nullable=False)
    
    # Relationship
    reviews = relationship("Review", back_populates="bank")

    def __repr__(self):
        return f"<Bank(name='{self.name}', short_name='{self.short_name}')>"

class Review(Base):
    """Model for storing bank app reviews."""
    __tablename__ = 'reviews'

    id = Column(Integer, primary_key=True)
    review_text = Column(Text, nullable=False)
    rating = Column(Integer, nullable=False)
    review_date = Column(Date, nullable=False)
    source = Column(String(50), nullable=False)
    app_version = Column(String(20))
    thumbs_up_count = Column(Integer, default=0)
    language = Column(String(10), nullable=False)
    country = Column(String(10), nullable=False)
    
    # Foreign Key
    bank_id = Column(Integer, ForeignKey('banks.id'), nullable=False)
    
    # Relationship
    bank = relationship("Bank", back_populates="reviews")

    def __repr__(self):
        return f"<Review(rating={self.rating}, date='{self.review_date}')>" 